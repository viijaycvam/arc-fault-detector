#ifndef TINYML_ENGINE_H
#define TINYML_ENGINE_H

#include <math.h>
#include "../reports/model_weights.h"

static inline int tinyml_predict(const float input_features[14], float *output_prob) {
    float scaled[14];
    float h1[16];
    float h2[8];
    float out_raw = 0.0f;

    // 1. StandardScaler Normalization
    for (int i = 0; i < 14; i++) {
        scaled[i] = (input_features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }

    // 2. Layer 1: 14 -> 16 (ReLU)
    for (int j = 0; j < 16; j++) {
        float sum = B1[j];
        for (int i = 0; i < 14; i++) {
            sum += scaled[i] * W1[i][j];
        }
        h1[j] = sum > 0.0f ? sum : 0.0f;
    }

    // 3. Layer 2: 16 -> 8 (ReLU)
    for (int j = 0; j < 8; j++) {
        float sum = B3[j];
        for (int i = 0; i < 16; i++) {
            sum += h1[i] * W2[i][j];
        }
        h2[j] = sum > 0.0f ? sum : 0.0f;
    }

    // 4. Output Layer: 8 -> 1
    out_raw = B3_OUT[0];
    for (int i = 0; i < 8; i++) {
        out_raw += h2[i] * W3[i][0];
    }

    // 5. Sigmoid Activation
    float prob = 1.0f / (1.0f + expf(-out_raw));
    if (output_prob != NULL) {
        *output_prob = prob;
    }

    return (prob >= 0.5f) ? 1 : 0;
}

#endif // TINYML_ENGINE_H