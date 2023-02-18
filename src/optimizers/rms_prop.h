#pragma once

#include <stdbool.h>
#include <stdint.h>

typedef struct rms_prop
{
    float learning_rate;
    float momentum;
    float ep;
    float t;

    float* s_dvar;

    bool is_built;
} rms_prop;

rms_prop* rms_prop_create(float learning_rate, float momentum, float ep);
void rms_prop_destroy(rms_prop* rmsp);

void rms_prop_apply_gradients(rms_prop* rmsp, float* grads, float* vars, uint32_t grads_vars_count);