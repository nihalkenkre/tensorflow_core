#pragma once

#include <stdint.h>
#include <stdbool.h>

typedef struct momentum
{
    float learning_rate;
    float momentum;
    float t;
    float* v_dvar;

    bool is_built;
} momentum;

momentum* momentum_create(float learning_rate, float beta);
void momentum_destroy(momentum* m);

void momentum_apply_gradients(momentum* m, float* grads, float* vars, uint32_t grads_vars_count);