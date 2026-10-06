#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "model_weights_b.h"
#include "golden_cycles.h"
#include "feature_extraction.h"

static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

float predict_mlp_c(const float features[8]) {
    float scaled[8], h1[16], h2[8], out_raw = 0.0f;

    for (int i = 0; i < 8; i++) {
        scaled[i] = (features[i] - SCALER_MEAN_B[i]) / SCALER_SCALE_B[i];
    }
    for (int j = 0; j < 16; j++) {
        float sum = B1_B[j];
        for (int i = 0; i < 8; i++) sum += scaled[i] * W1_B[i][j];
        h1[j] = relu(sum);
    }
    for (int j = 0; j < 8; j++) {
        float sum = B2_B[j];
        for (int i = 0; i < 16; i++) sum += h1[i] * W2_B[i][j];
        h2[j] = relu(sum);
    }
    out_raw = B3_B[0];
    for (int i = 0; i < 8; i++) out_raw += h2[i] * W3_B[i][0];

    return sigmoid(out_raw);
}

int main(void) {
    printf("====================================================\n");
    printf(" ADVANCED DSP FEATURE & INFERENCE DIAGNOSTICS\n");
    printf("====================================================\n\n");

    const char *feat_names[8] = {
        "crest", "skew", "kurt", "d1_ratio",
        "d2_ratio", "peak_asym", "energy_asym", "frac_small"
    };

    float max_diff[8] = {0.0f};
    int feat_pass_count[8] = {0};
    int full_feature_pass = 0;
    int mlp_pass = 0;
    const float TOLERANCE = 1e-2f;

    for (int i = 0; i < NUM_GOLDEN_CYCLES; i++) {
        const GoldenCycle *gc = &GOLDEN_CYCLES[i];
        float computed[8];

        extract_features_8(gc->raw_samples, computed);

        int all_feat_ok = 1;
        for (int f = 0; f < 8; f++) {
            float diff = fabsf(computed[f] - gc->expected_features[f]);
            if (diff > max_diff[f]) {
                max_diff[f] = diff;
            }
            if (diff <= TOLERANCE) {
                feat_pass_count[f]++;
            } else {
                all_feat_ok = 0;
            }
        }
        if (all_feat_ok) full_feature_pass++;

        float prob_c = predict_mlp_c(computed);
        int pred_c = (prob_c >= 0.5f) ? 1 : 0;
        int pred_py = (gc->expected_prob >= 0.5f) ? 1 : 0;

        if (pred_c == pred_py) {
            mlp_pass++;
        } else {
            printf("[MISMATCH] Cycle %2d (ID %4d) | Py Prob: %.4f | C Prob: %.4f\n",
                   i, gc->record_id, gc->expected_prob, prob_c);
            printf("  Feature breakdown for Cycle %d:\n", i);
            for (int f = 0; f < 8; f++) {
                printf("    %-11s -> C: %12.6f | Py: %12.6f | Diff: %12.6e\n",
                       feat_names[f], computed[f], gc->expected_features[f],
                       fabsf(computed[f] - gc->expected_features[f]));
            }
            printf("\n");
        }
    }

    printf("\n%-12s | %-18s | %-15s\n", "Feature", "Max Abs Diff", "Pass Rate (<=1e-2)");
    printf("---------------------------------------------------\n");
    for (int f = 0; f < 8; f++) {
        printf("%-12s | %18.8e | %d / %d\n", feat_names[f], max_diff[f], feat_pass_count[f], NUM_GOLDEN_CYCLES);
    }

    printf("\n---------------------------------------------------\n");
    printf("All-8 Features Match Cycles : %d / %d\n", full_feature_pass, NUM_GOLDEN_CYCLES);
    printf("Full Pipeline Inference Match: %d / %d\n", mlp_pass, NUM_GOLDEN_CYCLES);
    printf("---------------------------------------------------\n");

    return 0;
}