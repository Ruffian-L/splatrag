#include "dream_kernels.h"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kSemDim = 64;
constexpr int kMaxSemNbr = 32;
constexpr float kMinSeparation = 0.05f;
constexpr int kThreads = 256;

#define CUDA_RETURN(expr)                                                      \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr, "dream cuda: %s\n", cudaGetErrorString(_err));\
            return -1;                                                         \
        }                                                                      \
    } while (0)

__device__ __forceinline__ float sem_dot(const float *a, const float *b) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < kSemDim; ++k) {
        s += a[k] * b[k];
    }
    return s;
}

__device__ __forceinline__ uint64_t pack_cell(int x, int y, int z) {
    const uint32_t ux = static_cast<uint32_t>(x + (1 << 20));
    const uint32_t uy = static_cast<uint32_t>(y + (1 << 20));
    const uint32_t uz = static_cast<uint32_t>(z + (1 << 20));
    return (static_cast<uint64_t>(ux) << 42) | (static_cast<uint64_t>(uy) << 21) |
           static_cast<uint64_t>(uz);
}

__device__ __forceinline__ int3 cell_of(const float *pos, float size) {
    return make_int3(
        static_cast<int>(floorf(pos[0] / size)),
        static_cast<int>(floorf(pos[1] / size)),
        static_cast<int>(floorf(pos[2] / size))
    );
}

__device__ int lower_bound_keys(const uint64_t *keys, int n, uint64_t key) {
    int lo = 0;
    int hi = n;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (keys[mid] < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__device__ __forceinline__ bool in_sem_list(const int32_t *nbr, int other) {
#pragma unroll
    for (int k = 0; k < kMaxSemNbr; ++k) {
        if (nbr[k] == other) {
            return true;
        }
        if (nbr[k] < 0) {
            return false;
        }
    }
    return false;
}

__device__ void apply_pair(
    float *force,
    const float *pos_i,
    const float *pos_j,
    float mass_i,
    float mass_j,
    uint32_t dom_i,
    uint32_t dom_j,
    float cosine_dot,
    bool allow_attract,
    const DreamCudaParams p
) {
    float dx = pos_j[0] - pos_i[0];
    float dy = pos_j[1] - pos_i[1];
    float dz = pos_j[2] - pos_i[2];
    float distance = sqrtf(dx * dx + dy * dy + dz * dz);
    if (distance < kMinSeparation) {
        distance = kMinSeparation;
    }

    float weight = cosine_dot - p.semantic_threshold;
    if (dom_i != dom_j) {
        weight -= p.cross_domain_repulsion;
    }
    if (mass_i < 0.f || mass_j < 0.f) {
        weight = fminf(weight, -0.05f);
    }

    if (weight > 0.f) {
        if (allow_attract) {
            const float s = p.attraction * weight;
            force[0] += dx * s;
            force[1] += dy * s;
            force[2] += dz * s;
        }
    } else {
        const float falloff = 1.f + distance * distance;
        const float strength = p.semantic_repulsion * (-weight) / falloff;
        const float s = -strength / distance;
        force[0] += dx * s;
        force[1] += dy * s;
        force[2] += dz * s;
    }

    if (distance < p.repulsion_radius) {
        const float strength = (p.repulsion_radius - distance) * p.repulsion;
        const float s = -strength / distance;
        force[0] += dx * s;
        force[1] += dy * s;
        force[2] += dz * s;
    }
}

__global__ void fill_keys_kernel(
    int n,
    const float *pos,
    float cell_size,
    uint64_t *keys,
    int *idx
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int3 c = cell_of(pos + i * 3, cell_size);
    keys[i] = pack_cell(c.x, c.y, c.z);
    idx[i] = i;
}

__global__ void force_full_kernel(
    DreamCudaParams p,
    const float *pos,
    const float *mass,
    const uint32_t *domain,
    const float *sem,
    float *force
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) {
        return;
    }
    const float *pi = pos + i * 3;
    float f[3] = {
        -pi[0] * p.origin_pull,
        -pi[1] * p.origin_pull,
        -pi[2] * p.origin_pull,
    };
    const float mi = mass[i];
    const uint32_t di = domain[i];
    const float *si = sem + i * kSemDim;
    for (int j = 0; j < p.n; ++j) {
        if (j == i) {
            continue;
        }
        apply_pair(
            f,
            pi,
            pos + j * 3,
            mi,
            mass[j],
            di,
            domain[j],
            sem_dot(si, sem + j * kSemDim),
            true,
            p
        );
    }
    force[i * 3 + 0] = f[0];
    force[i * 3 + 1] = f[1];
    force[i * 3 + 2] = f[2];
}

__global__ void force_neighbor_kernel(
    DreamCudaParams p,
    const float *pos,
    const float *mass,
    const uint32_t *domain,
    const float *sem,
    const int32_t *sem_nbr,
    const float *sem_w,
    const uint64_t *keys_sorted,
    const int *idx_sorted,
    float *force
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) {
        return;
    }
    const float *pi = pos + i * 3;
    float f[3] = {
        -pi[0] * p.origin_pull,
        -pi[1] * p.origin_pull,
        -pi[2] * p.origin_pull,
    };
    const float mi = mass[i];
    const uint32_t di = domain[i];
    const int32_t *nbr = sem_nbr + i * kMaxSemNbr;
    const float *wts = sem_w + i * kMaxSemNbr;

    for (int k = 0; k < kMaxSemNbr; ++k) {
        const int j = nbr[k];
        if (j < 0) {
            break;
        }
        if (j == i) {
            continue;
        }
        apply_pair(
            f,
            pi,
            pos + j * 3,
            mi,
            mass[j],
            di,
            domain[j],
            wts[k],
            true,
            p
        );
    }

    const float cell = fmaxf(p.neighbor_radius, 0.001f);
    const int3 origin = cell_of(pi, cell);
    const float radius = p.neighbor_radius;
    const float radius2 = radius * radius;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                const uint64_t key = pack_cell(origin.x + dx, origin.y + dy, origin.z + dz);
                int cursor = lower_bound_keys(keys_sorted, p.n, key);
                while (cursor < p.n && keys_sorted[cursor] == key) {
                    const int j = idx_sorted[cursor++];
                    if (j == i || in_sem_list(nbr, j)) {
                        continue;
                    }
                    const float *pj = pos + j * 3;
                    const float ex = pj[0] - pi[0];
                    const float ey = pj[1] - pi[1];
                    const float ez = pj[2] - pi[2];
                    if (ex * ex + ey * ey + ez * ez > radius2) {
                        continue;
                    }
                    apply_pair(
                        f,
                        pi,
                        pj,
                        mi,
                        mass[j],
                        di,
                        domain[j],
                        sem_dot(sem + i * kSemDim, sem + j * kSemDim),
                        false,
                        p
                    );
                }
            }
        }
    }

    force[i * 3 + 0] = f[0];
    force[i * 3 + 1] = f[1];
    force[i * 3 + 2] = f[2];
}

__global__ void integrate_kernel(
    DreamCudaParams p,
    const float *force,
    const float *mass,
    float *pos,
    float *vel,
    float *disp_out,
    float *energy_out
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) {
        return;
    }
    const float inertia = fmaxf(fabsf(mass[i]), 0.1f);
    float v0 = vel[i * 3 + 0];
    float v1 = vel[i * 3 + 1];
    float v2 = vel[i * 3 + 2];
    v0 = (v0 + (force[i * 3 + 0] / inertia) * p.dt) * p.damping;
    v1 = (v1 + (force[i * 3 + 1] / inertia) * p.dt) * p.damping;
    v2 = (v2 + (force[i * 3 + 2] / inertia) * p.dt) * p.damping;
    const float d0 = v0 * p.dt;
    const float d1 = v1 * p.dt;
    const float d2 = v2 * p.dt;
    pos[i * 3 + 0] += d0;
    pos[i * 3 + 1] += d1;
    pos[i * 3 + 2] += d2;
    vel[i * 3 + 0] = v0;
    vel[i * 3 + 1] = v1;
    vel[i * 3 + 2] = v2;
    atomicAdd(disp_out, fabsf(d0) + fabsf(d1) + fabsf(d2));
    atomicAdd(energy_out, 0.5f * inertia * (v0 * v0 + v1 * v1 + v2 * v2));
}

int blocks_for(int n) {
    return (n + kThreads - 1) / kThreads;
}

} // namespace

struct DreamCuda {
    DreamCudaParams p{};
    float *pos = nullptr;
    float *vel = nullptr;
    float *mass = nullptr;
    uint32_t *domain = nullptr;
    float *sem = nullptr;
    int32_t *sem_nbr = nullptr;
    float *sem_w = nullptr;
    float *force = nullptr;
    uint64_t *keys = nullptr;
    uint64_t *keys_sorted = nullptr;
    int *idx = nullptr;
    int *idx_sorted = nullptr;
    void *cub_tmp = nullptr;
    size_t cub_bytes = 0;
    float *disp = nullptr;
    float *energy = nullptr;
};

extern "C" int dream_cuda_probe(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        return -1;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        return -1;
    }
    void *probe = nullptr;
    if (cudaMalloc(&probe, 4) != cudaSuccess) {
        return -1;
    }
    cudaFree(probe);
    return 0;
}

extern "C" DreamCuda *dream_cuda_create(
    const DreamCudaParams *p,
    const float *pos,
    const float *vel,
    const float *mass,
    const uint32_t *domain,
    const float *semantics,
    const int32_t *sem_nbr,
    const float *sem_w
) {
    if (!p || p->n <= 0) {
        return nullptr;
    }
    if (dream_cuda_probe() != 0) {
        return nullptr;
    }
    auto *d = new DreamCuda();
    d->p = *p;
    const int n = p->n;
    auto fail = [&]() -> DreamCuda * {
        dream_cuda_destroy(d);
        return nullptr;
    };
    if (cudaMalloc(&d->pos, sizeof(float) * n * 3) != cudaSuccess) return fail();
    if (cudaMalloc(&d->vel, sizeof(float) * n * 3) != cudaSuccess) return fail();
    if (cudaMalloc(&d->mass, sizeof(float) * n) != cudaSuccess) return fail();
    if (cudaMalloc(&d->domain, sizeof(uint32_t) * n) != cudaSuccess) return fail();
    if (cudaMalloc(&d->sem, sizeof(float) * n * kSemDim) != cudaSuccess) return fail();
    if (cudaMalloc(&d->sem_nbr, sizeof(int32_t) * n * kMaxSemNbr) != cudaSuccess) return fail();
    if (cudaMalloc(&d->sem_w, sizeof(float) * n * kMaxSemNbr) != cudaSuccess) return fail();
    if (cudaMalloc(&d->force, sizeof(float) * n * 3) != cudaSuccess) return fail();
    if (cudaMalloc(&d->keys, sizeof(uint64_t) * n) != cudaSuccess) return fail();
    if (cudaMalloc(&d->keys_sorted, sizeof(uint64_t) * n) != cudaSuccess) return fail();
    if (cudaMalloc(&d->idx, sizeof(int) * n) != cudaSuccess) return fail();
    if (cudaMalloc(&d->idx_sorted, sizeof(int) * n) != cudaSuccess) return fail();
    if (cudaMalloc(&d->disp, sizeof(float)) != cudaSuccess) return fail();
    if (cudaMalloc(&d->energy, sizeof(float)) != cudaSuccess) return fail();

    cub::DeviceRadixSort::SortPairs(
        nullptr,
        d->cub_bytes,
        d->keys,
        d->keys_sorted,
        d->idx,
        d->idx_sorted,
        n
    );
    if (d->cub_bytes > 0) {
        if (cudaMalloc(&d->cub_tmp, d->cub_bytes) != cudaSuccess) return fail();
    }

    if (cudaMemcpy(d->pos, pos, sizeof(float) * n * 3, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail();
    if (cudaMemcpy(d->vel, vel, sizeof(float) * n * 3, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail();
    if (cudaMemcpy(d->mass, mass, sizeof(float) * n, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail();
    if (cudaMemcpy(d->domain, domain, sizeof(uint32_t) * n, cudaMemcpyHostToDevice) != cudaSuccess)
        return fail();
    if (cudaMemcpy(d->sem, semantics, sizeof(float) * n * kSemDim, cudaMemcpyHostToDevice) !=
        cudaSuccess)
        return fail();
    if (cudaMemcpy(d->sem_nbr, sem_nbr, sizeof(int32_t) * n * kMaxSemNbr, cudaMemcpyHostToDevice) !=
        cudaSuccess)
        return fail();
    if (cudaMemcpy(d->sem_w, sem_w, sizeof(float) * n * kMaxSemNbr, cudaMemcpyHostToDevice) !=
        cudaSuccess)
        return fail();
    return d;
}

extern "C" int dream_cuda_step(DreamCuda *d, float *out_disp, float *out_energy) {
    if (!d || !out_disp || !out_energy) {
        return -1;
    }
    const int n = d->p.n;
    const int blocks = blocks_for(n);
    CUDA_RETURN(cudaMemset(d->disp, 0, sizeof(float)));
    CUDA_RETURN(cudaMemset(d->energy, 0, sizeof(float)));

    if (d->p.neighbor_mode) {
        fill_keys_kernel<<<blocks, kThreads>>>(
            n,
            d->pos,
            fmaxf(d->p.neighbor_radius, 0.001f),
            d->keys,
            d->idx
        );
        CUDA_RETURN(cudaGetLastError());
        cub::DeviceRadixSort::SortPairs(
            d->cub_tmp,
            d->cub_bytes,
            d->keys,
            d->keys_sorted,
            d->idx,
            d->idx_sorted,
            n
        );
        CUDA_RETURN(cudaGetLastError());
        force_neighbor_kernel<<<blocks, kThreads>>>(
            d->p,
            d->pos,
            d->mass,
            d->domain,
            d->sem,
            d->sem_nbr,
            d->sem_w,
            d->keys_sorted,
            d->idx_sorted,
            d->force
        );
    } else {
        force_full_kernel<<<blocks, kThreads>>>(
            d->p,
            d->pos,
            d->mass,
            d->domain,
            d->sem,
            d->force
        );
    }
    CUDA_RETURN(cudaGetLastError());
    integrate_kernel<<<blocks, kThreads>>>(
        d->p,
        d->force,
        d->mass,
        d->pos,
        d->vel,
        d->disp,
        d->energy
    );
    CUDA_RETURN(cudaGetLastError());
    CUDA_RETURN(cudaDeviceSynchronize());
    CUDA_RETURN(cudaMemcpy(out_disp, d->disp, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_RETURN(cudaMemcpy(out_energy, d->energy, sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" int dream_cuda_copy_state(DreamCuda *d, float *pos, float *vel) {
    if (!d || !pos || !vel) {
        return -1;
    }
    const int n = d->p.n;
    CUDA_RETURN(cudaMemcpy(pos, d->pos, sizeof(float) * n * 3, cudaMemcpyDeviceToHost));
    CUDA_RETURN(cudaMemcpy(vel, d->vel, sizeof(float) * n * 3, cudaMemcpyDeviceToHost));
    return 0;
}

extern "C" void dream_cuda_destroy(DreamCuda *d) {
    if (!d) {
        return;
    }
    cudaFree(d->pos);
    cudaFree(d->vel);
    cudaFree(d->mass);
    cudaFree(d->domain);
    cudaFree(d->sem);
    cudaFree(d->sem_nbr);
    cudaFree(d->sem_w);
    cudaFree(d->force);
    cudaFree(d->keys);
    cudaFree(d->keys_sorted);
    cudaFree(d->idx);
    cudaFree(d->idx_sorted);
    cudaFree(d->cub_tmp);
    cudaFree(d->disp);
    cudaFree(d->energy);
    delete d;
}
