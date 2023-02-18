#pragma once

#include <stdint.h>

typedef struct gradient_descent
{
    float learning_rate;
} gradient_descent;

gradient_descent *gradient_descent_create(float learning_rate);
void gradient_descent_destroy(gradient_descent *gd);

void gradient_descent_apply_gradients(gradient_descent *gd, float *grads, float *vars, uint32_t grads_vars_count);