#include "momentum.h"
#include <stdlib.h>

momentum *momentum_create(float learning_rate, float beta)
{
    momentum *m = (momentum *)calloc(1, sizeof(momentum));

    m->learning_rate = learning_rate;
    m->momentum = beta;
    m->t = 1.f;
    m->is_built = false;

    return m;
}

void momentum_destroy(momentum *m)
{
    if (m != NULL)
    {
        if (m->v_dvar != NULL)
        {
            free(m->v_dvar);
        }

        free(m);
    }
}

void momentum_apply_gradients(momentum *m, float *grads, float *vars, uint32_t grads_vars_count)
{
    if (!m->is_built)
    {
        m->v_dvar = (float *)calloc(grads_vars_count, sizeof(float));
        m->is_built = true;
    }

    for (uint32_t gv = 0; gv < grads_vars_count; ++gv)
    {
        m->v_dvar[gv] = (m->momentum * m->v_dvar[gv]) + ((1 - m->momentum) * grads[gv]);
        vars[gv] -= m->learning_rate * m->v_dvar[gv];
    }

    ++m->t;
}