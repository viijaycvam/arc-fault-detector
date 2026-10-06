#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "model_weights_b.h"
#include "golden_cycles.h"

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float predict_mlp_c(const float raw_features[8]) {
    float scaled[8];
    float h1[16];
    float h2[8];
    float out_raw = 0.0f;

    for (int i = 0; i < 8; i++) {
        scaled[i] = (raw_features[i] - SCALER_MEAN_B[i]) / SCALER_SCALE_B[i];
    }

    for (int j = 0; j < 16; j++) {
        float sum = B1_B[j];
        for (int i = 0; i < 8; i++) {
            sum += scaled[i] * W1_B[i][j];
        }
        h1[j] = relu(sum);
    }

    for (int j = 0; j < 8; j++) {
        float sum = B2_B[j];
        for (int i = 0; i < 16; i++) {
            sum += h1[i] * W2_B[i][j];
        }
        h2[j] = relu(sum);
    }

    out_raw = B3_B[0];
    for (int i = 0; i < 8; i++) {
        out_raw += h2[i] * W3_B[i][0];
    }

    return sigmoid(out_raw);
}

int main(void) {
    printf("====================================================\n");
    printf(" C INFERENCE PARITY VERIFICATION (C vs PYTHON)\n");
    printf("====================================================\n\n");

    int match_count = 0;
    float max_prob_error = 0.0f;
    const float TOLERANCE = 1e-4f;

    for (int i = 0; i < NUM_GOLDEN_CYCLES; i++) {
        const GoldenCycle *gc = &GOLDEN_CYCLES[i];

        float prob_c = predict_mlp_c(gc->expected_features);
        int pred_c = (prob_c >= 0.5f) ? 1 : 0;
        int pred_py = (gc->expected_prob >= 0.5f) ? 1 : 0;
        float err = fabsf(prob_c - gc->expected_prob);

        if (err > max_prob_error) {
            max_prob_error = err;
        }

        if (pred_c == pred_py && err < TOLERANCE) {
            match_count++;
        } else {
            printf("[MISMATCH] Record ID %4d | Python Prob: %.6f | C Prob: %.6f | Error: %.2e\n",
                   gc->record_id, gc->expected_prob, prob_c, err);
        }
    }

    printf("\n----------------------------------------------------\n");
    printf("Matching Cycles : %d / %d\n", match_count, NUM_GOLDEN_CYCLES);
    printf("Max Prob Error  : %.8e\n", max_prob_error);
    printf("----------------------------------------------------\n");

    if (match_count == NUM_GOLDEN_CYCLES) {
        printf("STATUS: PARITY VERIFICATION SUCCESSFUL (100%% match with Python)\n");
    } else {
        printf("STATUS: VERIFICATION FAILED\n");
    }

    return 0;
}