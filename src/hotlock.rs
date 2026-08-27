//! Write-scoped single-writer guard for hot state.
//!
//! Hot state was guarded only by an in-process `Mutex`, which says nothing across
//! processes. Seven `splatrag ... mcp` processes each held their own copy and each
//! persisted it, so the last write won regardless of what it knew. Measured: a dream
//! wrote `dream_cycle = 16` with 34,040 `muted` flags, and a process holding a
//! two-hour-old snapshot wrote `dream_cycle = 15` and no flags back over it.
//!
//! The first fix held the lock for process lifetime. That killed the clobber and broke
//! the contract: ingest-everything runs `splatrag ingest` from silo watchers, and a
//! long-lived MCP that won the startup race would starve every one of them. So the
//! lock is now held only around a write, and staleness is caught separately:
//!
//! * **Lock** serialises writers — one at a time, briefly.
//! * **`dream_cycle` guard** catches the actual clobber class — a snapshot loaded
//!   before someone else's dream trying to overwrite it.
//! * **Inode + pid recheck on release** closes the hole left by a lock file removed
//!   underneath its holder: ownership of a *path* is not ownership of a *file*.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

/// Exclusive right to write hot state, for the duration of one write.
#[derive(Debug)]
pub struct WriteGuard {
    path: PathBuf,
    pid: u32,
    ino: u64,
    /// Unique per acquisition. Inode numbers alone are not enough: a freed inode is
    /// routinely handed straight back out, so a lock removed and retaken can land on
    /// the same `ino` and a stale guard would still believe it owns the file. Caught
    /// by `lock_deleted_underneath_holder_loses_ownership`.
    token: String,
    released: bool,
}

static ACQUIRE_SEQ: AtomicU64 = AtomicU64::new(0);

fn mint_token(pid: u32) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = ACQUIRE_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("{pid}-{nanos}-{seq}")
}

impl WriteGuard {
    /// Take the write lock, waiting up to `timeout` for a current writer to finish.
    pub fn acquire(hot_dir: &Path, timeout: Duration) -> Result<Self> {
        let path = hot_dir.join(".writer.lock");
        fs::create_dir_all(hot_dir).ok();
        let pid = std::process::id();
        let deadline = Instant::now() + timeout;
        loop {
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    let token = mint_token(pid);
                    writeln!(file, "{pid}")?;
                    writeln!(file, "{token}")?;
                    file.sync_all().ok();
                    let ino = fs::metadata(&path)?.ino();
                    return Ok(Self {
                        path,
                        pid,
                        ino,
                        token,
                        released: false,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    // Reclaim only from an owner that is provably gone.
                    if let Some(owner) = read_owner(&path) {
                        if !process_alive(owner) {
                            let _ = fs::remove_file(&path);
                            continue;
                        }
                    } else {
                        // Unreadable/empty lock with no parseable owner: treat as debris.
                        let _ = fs::remove_file(&path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        anyhow::bail!(
                            "timed out after {:?} waiting for the hot-state write lock (owner {:?})",
                            timeout,
                            read_owner(&path)
                        );
                    }
                    thread::sleep(Duration::from_millis(50));
                }
                Err(error) => return Err(error).context("opening hot-state write lock"),
            }
        }
    }

    /// Do we still hold *this file*, not merely this path?
    ///
    /// A lock file deleted underneath us (or replaced by another writer) means our
    /// ownership is gone even though the process never released it. Writing after that
    /// is exactly the unserialised write the lock exists to prevent.
    pub fn still_owned(&self) -> bool {
        match fs::metadata(&self.path) {
            // Inode is the cheap gate; the token is what actually decides, because
            // inode numbers are recycled.
            Ok(meta) => {
                meta.ino() == self.ino
                    && read_owner(&self.path) == Some(self.pid)
                    && read_token(&self.path).as_deref() == Some(self.token.as_str())
            }
            Err(_) => false,
        }
    }

    /// Release explicitly, refusing to delete a file that is no longer ours.
    pub fn release(&mut self) {
        if !self.released && self.still_owned() {
            let _ = fs::remove_file(&self.path);
        }
        self.released = true;
    }
}

impl Drop for WriteGuard {
    fn drop(&mut self) {
        self.release();
    }
}

fn read_owner(path: &Path) -> Option<u32> {
    fs::read_to_string(path)
        .ok()?
        .trim()
        .lines()
        .next()?
        .parse()
        .ok()
}

fn read_token(path: &Path) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    Some(text.lines().nth(1)?.trim().to_string())
}

fn process_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// Read `dream_cycle` from an on-disk hot state without parsing 95 MB of splats.
///
/// The field is emitted second, so the head of the file is enough.
pub fn peek_dream_cycle(path: &Path) -> Option<u64> {
    let mut file = fs::File::open(path).ok()?;
    let mut head = vec![0u8; 4096];
    let n = file.read(&mut head).ok()?;
    let text = String::from_utf8_lossy(&head[..n]);
    let at = text.find("\"dream_cycle\"")?;
    let rest = &text[at + "\"dream_cycle\"".len()..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit())?;
    rest[..end].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("splatrag-hotlock-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn second_writer_waits_then_times_out() {
        let dir = tmp("serialise");
        let _held = WriteGuard::acquire(&dir, Duration::from_millis(50)).unwrap();
        // A live owner must not be reclaimed: the second writer waits and gives up.
        let second = WriteGuard::acquire(&dir, Duration::from_millis(120));
        assert!(second.is_err(), "two writers held the lock at once");
    }

    #[test]
    fn released_lock_is_immediately_reacquirable() {
        let dir = tmp("cycle");
        {
            let _g = WriteGuard::acquire(&dir, Duration::from_secs(1)).unwrap();
        }
        WriteGuard::acquire(&dir, Duration::from_millis(50))
            .expect("lock should be free once the guard dropped");
    }

    #[test]
    fn lock_deleted_underneath_holder_loses_ownership() {
        let dir = tmp("stolen");
        let mut guard = WriteGuard::acquire(&dir, Duration::from_secs(1)).unwrap();
        assert!(guard.still_owned());

        // The `rm -f` hole: the path is gone, so this process no longer owns anything.
        fs::remove_file(dir.join(".writer.lock")).unwrap();
        assert!(
            !guard.still_owned(),
            "holder must notice its lock file vanished"
        );

        // Another writer takes the path; releasing must not delete their lock.
        let other = WriteGuard::acquire(&dir, Duration::from_secs(1)).unwrap();
        guard.release();
        assert!(
            other.still_owned(),
            "release deleted a lock owned by someone else"
        );
    }

    #[test]
    fn replaced_lock_file_loses_ownership() {
        let dir = tmp("replaced");
        let guard = WriteGuard::acquire(&dir, Duration::from_secs(1)).unwrap();
        let path = dir.join(".writer.lock");
        fs::remove_file(&path).unwrap();
        // Same path, different inode and a different owner pid.
        fs::write(&path, format!("{}\nsomeone-else\n", guard.pid + 1)).unwrap();
        assert!(
            !guard.still_owned(),
            "inode/pid change must revoke ownership"
        );
    }

    #[test]
    fn dead_owner_is_reclaimed() {
        let dir = tmp("dead");
        // pid 1 exists; use an implausible one that is not running.
        fs::write(dir.join(".writer.lock"), "4294967294\nstale-token\n").unwrap();
        WriteGuard::acquire(&dir, Duration::from_millis(200))
            .expect("a lock whose owner is gone must be reclaimable");
    }

    #[test]
    fn peek_reads_dream_cycle_without_the_splats() {
        let dir = tmp("peek");
        let path = dir.join("state.json");
        let big = "0.123456,".repeat(2000);
        fs::write(
            &path,
            format!(
                "{{\"version\":1,\"dream_cycle\":16,\"kinetic_energy\":1.0,\"splats\":[{big}]}}"
            ),
        )
        .unwrap();
        assert_eq!(peek_dream_cycle(&path), Some(16));
    }
}
