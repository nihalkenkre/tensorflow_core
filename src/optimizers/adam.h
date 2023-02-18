#pragma once

#include <stdint.h>
#include <stdbool.h>

typedef struct adam
{
    float learning_rate;
    float beta_1;
    float beta_2;
    float ep;

    float *v_dvar;
    float *s_dvar;

    float t;

    bool is_built;
} adam;

adam* adam_create(float learning_rate, float beta_1, float beta_2, float ep);
void adam_destroy(adam* a);

void adam_apply_gradients(adam* a, float* grads, float* vars, uint32_t grads_vars_count);