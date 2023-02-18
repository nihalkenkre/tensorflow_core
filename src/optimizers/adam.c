#include "adam.h"

#include <stdlib.h>
#include <math.h>

adam *adam_create(float learning_rate, float beta_1, float beta_2, float ep)
{
    adam *a = (adam *)calloc(1, sizeof(adam));

    a->learning_rate = learning_rate;
    a->beta_1 = beta_1;
    a->beta_2 = beta_2;
    a->ep = ep;

    a->t = 1.f;

    a->is_built = false;

    return a;
}

void adam_destroy(adam *a)
{
    if (a != NULL)
    {
        if (a->v_dvar != NULL)
        {
            free(a->v_dvar);
        }

        if (a->s_dvar != NULL)
        {
            free(a->s_dvar);
        }

        free(a);
    }
}

void adam_apply_gradients(adam *a, float *grads, float *vars, uint32_t grads_vars_count)
{
    if (!a->is_built)
    {
        a->v_dvar = (float *)calloc(grads_vars_count, sizeof(float));
        a->s_dvar = (float *)calloc(grads_vars_count, sizeof(float));

        a->is_built = true;
    }

    for (uint32_t gv = 0; gv < grads_vars_count; ++gv)
    {
        a->v_dvar[gv] = (a->beta_1 * a->v_dvar[gv]) + ((1.f - a->beta_1) * grads[gv]);
        a->s_dvar[gv] = (a->beta_2 * a->s_dvar[gv]) + ((1.f - a->beta_2) * powf(grads[gv], 2));

        float v_dvar_bc = a->v_dvar[gv] / (1.f - powf(a->beta_1, a->t));
        float s_dvar_bc = a->s_dvar[gv] / (1.f - powf(a->beta_2, a->t));

        vars[gv] -= a->learning_rate * (v_dvar_bc / (sqrtf(s_dvar_bc) + a->ep));
    }

    ++a->t;
}