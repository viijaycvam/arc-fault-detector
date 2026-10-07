#ifndef TINYML_ENGINE_B_H
#define TINYML_ENGINE_B_H

#include <math.h>
#include "model_weights_b.h"

#define AFD_FEATURE_COUNT 8

static inline float afd_relu(float x)
{
    return (x > 0.0f) ? x : 0.0f;
}

static inline float afd_sigmoid(float x)
{
    if (x >= 40.0f) {
        return 1.0f;
    }

    if (x <= -40.0f) {
        return 0.0f;
    }

    return 1.0f / (1.0f + expf(-x));
}

/*
 * 8 -> 16 -> 8 -> 1 MLP
 *
 * Feature order:
 *   0 crest
 *   1 skew
 *   2 kurt
 *   3 d1_ratio
 *   4 d2_ratio
 *   5 peak_asym
 *   6 energy_asym
 *   7 frac_small
 */
static inline float predict_mlp_8(
    const float features[AFD_FEATURE_COUNT]
)
{
    float scaled[AFD_FEATURE_COUNT];
    float h1[16];
    float h2[8];

    /* StandardScaler */
    for (int i = 0; i < AFD_FEATURE_COUNT; i++) {
        scaled[i] =
            (features[i] - SCALER_MEAN_B[i])
            / SCALER_SCALE_B[i];
    }

    /* Layer 1: 8 -> 16 + ReLU */
    for (int j = 0; j < 16; j++) {
        float sum = B1_B[j];

        for (int i = 0; i < AFD_FEATURE_COUNT; i++) {
            sum += scaled[i] * W1_B[i][j];
        }

        h1[j] = afd_relu(sum);
    }

    /* Layer 2: 16 -> 8 + ReLU */
    for (int j = 0; j < 8; j++) {
        float sum = B2_B[j];

        for (int i = 0; i < 16; i++) {
            sum += h1[i] * W2_B[i][j];
        }

        h2[j] = afd_relu(sum);
    }

    /* Output layer: 8 -> 1 */
    float out_raw = B3_B[0];

    for (int i = 0; i < 8; i++) {
        out_raw += h2[i] * W3_B[i][0];
    }

    /* Sigmoid */
    return afd_sigmoid(out_raw);
}

/*
 * Returns:
 *   0 = NORMAL
 *   1 = ARC
 *
 * Also returns probability through the pointer.
 */
static inline int predict_arc_8(
    const float features[AFD_FEATURE_COUNT],
    float *probability
)
{
    float p = predict_mlp_8(features);

    if (probability != NULL) {
        *probability = p;
    }

    return (p >= 0.5f) ? 1 : 0;
}

#endif