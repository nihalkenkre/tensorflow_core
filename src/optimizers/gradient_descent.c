#include "gradient_descent.h"

#include <stdlib.h>

gradient_descent *gradient_descent_create(float learning_rate)
{
    gradient_descent *gd = (gradient_descent *)calloc(1, sizeof(gradient_descent));

    gd->learning_rate = learning_rate;

    return gd;
}

void gradient_descent_destroy(gradient_descent *gd)
{
    if (gd != NULL)
    {
        free(gd);
    }
}

void gradient_descent_apply_gradients(gradient_descent *gd, float *grads, float *vars,
                                      uint32_t grads_vars_count)
{
    for (uint32_t i = 0; i < grads_vars_count; ++i)
    {
        vars[i] -= (gd->learning_rate * grads[i]);
    }
}