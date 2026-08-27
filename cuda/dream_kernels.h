#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DreamCudaParams {
    int32_t n;
    int32_t neighbor_mode;
    float dt;
    float damping;
    float origin_pull;
    float attraction;
    float semantic_threshold;
    float repulsion_radius;
    float repulsion;
    float semantic_repulsion;
    float cross_domain_repulsion;
    float neighbor_radius;
} DreamCudaParams;

typedef struct DreamCuda DreamCuda;

/* Returns 0 if a CUDA device can be initialized. */
int dream_cuda_probe(void);

DreamCuda *dream_cuda_create(
    const DreamCudaParams *p,
    const float *pos,
    const float *vel,
    const float *mass,
    const uint32_t *domain,
    const float *semantics,
    const int32_t *sem_nbr,
    const float *sem_w
);

/* One integrate step. Writes total |Δx|+|Δy|+|Δz| and kinetic energy. 0 = ok. */
int dream_cuda_step(DreamCuda *d, float *out_disp, float *out_energy);

int dream_cuda_copy_state(DreamCuda *d, float *pos, float *vel);

void dream_cuda_destroy(DreamCuda *d);

#ifdef __cplusplus
}
#endif
