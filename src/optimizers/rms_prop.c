#include "rms_prop.h"

#include <stdlib.h>
#include <math.h>

rms_prop *rms_prop_create(float learning_rate, float momentum, float ep)
{
    rms_prop *rmsp = (rms_prop *)calloc(1, sizeof(rms_prop));

    rmsp->learning_rate = learning_rate;
    rmsp->momentum = momentum;
    rmsp->ep = ep;
    rmsp->t = 1.f;
    rmsp->is_built = false;

    return rmsp;
}

void rms_prop_destroy(rms_prop *rmsp)
{
    if (rmsp != NULL)
    {
        if (rmsp->s_dvar != NULL)
        {
            free(rmsp->s_dvar);
        }

        free(rmsp);
    }
}

void rms_prop_apply_gradients(rms_prop *rmsp, float *grads, float *vars, uint32_t grads_vars_count)
{
    if (!rmsp->is_built)
    {
        rmsp->s_dvar = (float *)calloc(grads_vars_count, sizeof(float));
        rmsp->is_built = true;
    }

    for (uint32_t gv = 0; gv < grads_vars_count; ++gv)
    {
        rmsp->s_dvar[gv] = (rmsp->momentum * rmsp->s_dvar[gv]) + ((1 - rmsp->momentum) * powf(grads[gv], 2));

        vars[gv] -= rmsp->learning_rate * (grads[gv] / (sqrtf(rmsp->s_dvar[gv]) + rmsp->ep));
    }

    ++rmsp->t;
}