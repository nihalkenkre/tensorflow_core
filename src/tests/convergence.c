#include <gradient_descent.h>
#include <momentum.h>
#include <rms_prop.h>
#include <adam.h>

#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

void convergence_gradient_decent(gradient_descent *gd, uint32_t epoch_count, float init_value, float *out_param_path)
{
    float x_current = init_value;
    bool has_converged = false;

    for (uint32_t i = 0; i < epoch_count; ++i)
    {
        // loss function = 2x^4 + 3x^3 + 2
        // derivative of loss function = 8x^3 + 9x^2
        float x_grad = (8 * powf(x_current, 3)) + (9 * powf(x_current, 2));

        if (isnan(x_grad))
        {
            printf("Gradient exploded after %d epochs for learning rate %f\n", i + 1, gd->learning_rate);
            break;
        }

        float x_old = x_current;

        gradient_descent_apply_gradients(gd, &x_grad, &x_current, 1);
        out_param_path[i] = x_current;

        if (x_current == x_old)
        {
            printf("Converged after %d epochs for learning rate %f\n", i + 1, gd->learning_rate);
            has_converged = true;

            break;
        }
    }

    if (!has_converged)
    {
        printf("Did not converge after %d epochs for learning rate %f\n", epoch_count, gd->learning_rate);
    }
}

void convergence_momentum(momentum *m, uint32_t epoch_count, float init_value, float *out_param_path)
{
    float x_current = init_value;
    bool has_converged = false;

    for (uint32_t i = 0; i < epoch_count; ++i)
    {
        // loss function = 2x^4 + 3x^3 + 2
        // derivative of loss function = 8x^3 + 9x^2
        float x_grad = (8 * powf(x_current, 3)) + (9 * powf(x_current, 2));

        if (isnan(x_grad))
        {
            printf("Gradient exploded after %d epochs for learning rate %f and momentum %f\n", i + 1, m->learning_rate, m->momentum);
            break;
        }

        float x_old = x_current;

        momentum_apply_gradients(m, &x_grad, &x_current, 1);
        out_param_path[i] = x_current;

        if (x_current == x_old)
        {
            printf("Converged after %d epochs for learning rate %f and momentum %f\n", i + 1, m->learning_rate, m->momentum);
            has_converged = true;

            break;
        }
    }

    if (!has_converged)
    {
        printf("Did not converge after %d epochs for learning rate %f and momentum %f\n", epoch_count, m->learning_rate, m->momentum);
    }
}

void convergence_rms_prop(rms_prop *rmsp, uint32_t epoch_count, float init_value, float *out_param_path)
{
    float x_current = init_value;
    bool has_converged = false;

    for (uint32_t i = 0; i < epoch_count; ++i)
    {
        // loss function = 2x^4 + 3x^3 + 2
        // derivative of loss function = 8x^3 + 9x^2
        float x_grad = (8 * powf(x_current, 3)) + (9 * powf(x_current, 2));

        if (isnan(x_grad))
        {
            printf("Gradient exploded after %d epochs for learning rate %f and momentum %f\n", i + 1, rmsp->learning_rate, rmsp->momentum);
            break;
        }

        float x_old = x_current;

        rms_prop_apply_gradients(rmsp, &x_grad, &x_current, 1);
        out_param_path[i] = x_current;

        if (x_current == x_old)
        {
            printf("Converged after %d epochs for learning rate %f and momentum %f\n", i + 1, rmsp->learning_rate, rmsp->momentum);
            has_converged = true;

            break;
        }
    }

    if (!has_converged)
    {
        printf("Did not converge after %d epochs for learning rate %f and momentum %f\n", epoch_count, rmsp->learning_rate, rmsp->momentum);
    }
}

void convergence_adam(adam *a, uint32_t epoch_count, float init_value, float *out_param_path)
{
    float x_current = init_value;
    bool has_converged = false;

    for (uint32_t i = 0; i < epoch_count; ++i)
    {
        // loss function = 2x^4 + 3x^3 + 2
        // derivative of loss function = 8x^3 + 9x^2
        float x_grad = (8 * powf(x_current, 3)) + (9 * powf(x_current, 2));

        if (isnan(x_grad))
        {
            printf("Gradient exploded after %d epochs for learning rate %f\n", i + 1, a->learning_rate);
            break;
        }

        float x_old = x_current;

        adam_apply_gradients(a, &x_grad, &x_current, 1);
        out_param_path[i] = x_current;

        if (x_current == x_old)
        {
            printf("Converged after %d epochs for learning rate %f\n", i + 1, a->learning_rate);
            has_converged = true;

            break;
        }
    }

    if (!has_converged)
    {
        printf("Did not converge after %d epochs for learning rate %f\n", epoch_count, a->learning_rate);
    }
}

void convergence_tests(void)
{
    float learning_rates[3] = {0.1f, 0.01f, 0.001f};
    float *out_param_path = (float *)calloc(2000, sizeof(float));

    printf("================\n");
    printf("Gradient Descent\n");
    printf("================\n");

    for (uint32_t lr = 0; lr < 3; ++lr)
    {
        gradient_descent *gd = gradient_descent_create(learning_rates[lr]);
        convergence_gradient_decent(gd, 2000, 2.f, out_param_path);
        gradient_descent_destroy(gd);
    }

    printf("========\n");
    printf("Momentum\n");
    printf("========\n");

    for (uint32_t lr = 0; lr < 3; ++lr)
    {
        momentum *m = momentum_create(learning_rates[lr], 0.9f);
        convergence_momentum(m, 2000, 2.f, out_param_path);
        momentum_destroy(m);
    }

    printf("========\n");
    printf("RMS Prop\n");
    printf("========\n");

    for (uint32_t lr = 0; lr < 3; ++lr)
    {
        rms_prop *rmsp = rms_prop_create(learning_rates[lr], 0.999f, (float)1e-7);
        convergence_rms_prop(rmsp, 2000, 2.f, out_param_path);
        rms_prop_destroy(rmsp);
    }

    printf("====\n");
    printf("Adam\n");
    printf("====\n");

    for (uint32_t lr = 0; lr < 3; ++lr)
    {
        adam *a = adam_create(learning_rates[lr], 0.9f, 0.999f, (float)1e-7);
        convergence_adam(a, 2000, 2.f, out_param_path);
        adam_destroy(a);
    }

    if (out_param_path != NULL)
    {
        free(out_param_path);
    }
}