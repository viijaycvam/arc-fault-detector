#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "model_weights_b.h"
#include "feature_extraction.h"

#define TEST_RECORDS 1334
#define FEATURE_COUNT 8
#define CYCLE_LEN 1000

#define FEATURE_TOLERANCE 1e-2f
#define PROB_TOLERANCE 1e-4f

static inline float relu(float x)
{
    return (x > 0.0f) ? x : 0.0f;
}

static float sigmoid(float x)
{
    if (x >= 40.0f) {
        return 1.0f;
    }

    if (x <= -40.0f) {
        return 0.0f;
    }

    return 1.0f / (1.0f + expf(-x));
}

static float predict_mlp_c(
    const float features[FEATURE_COUNT],
    float *out_probability
)
{
    float scaled[FEATURE_COUNT];
    float h1[16];
    float h2[8];

    /* StandardScaler */
    for (int i = 0; i < FEATURE_COUNT; i++) {
        scaled[i] =
            (features[i] - SCALER_MEAN_B[i])
            / SCALER_SCALE_B[i];
    }

    /* Layer 1: 8 -> 16 -> ReLU */
    for (int j = 0; j < 16; j++) {

        float sum = B1_B[j];

        for (int i = 0; i < FEATURE_COUNT; i++) {
            sum += scaled[i] * W1_B[i][j];
        }

        h1[j] = relu(sum);
    }

    /* Layer 2: 16 -> 8 -> ReLU */
    for (int j = 0; j < 8; j++) {

        float sum = B2_B[j];

        for (int i = 0; i < 16; i++) {
            sum += h1[i] * W2_B[i][j];
        }

        h2[j] = relu(sum);
    }

    /* Output: 8 -> 1 */
    float out_raw = B3_B[0];

    for (int i = 0; i < 8; i++) {
        out_raw += h2[i] * W3_B[i][0];
    }

    float probability = sigmoid(out_raw);

    if (out_probability != NULL) {
        *out_probability = probability;
    }

    return probability;
}

int main(void)
{
    printf("====================================================\n");
    printf("FULL OFFICIAL TEST SET - C/PYTHON EQUIVALENCE\n");
    printf("====================================================\n");

    FILE *raw_file =
        fopen("reports/full_test_raw.bin", "rb");

    FILE *feat_file =
        fopen("reports/full_test_features.bin", "rb");

    FILE *prob_file =
        fopen("reports/full_test_prob.bin", "rb");

    FILE *label_file =
        fopen("reports/full_test_labels.bin", "rb");

    FILE *pred_file =
        fopen("reports/full_test_pred.bin", "rb");

    if (!raw_file ||
        !feat_file ||
        !prob_file ||
        !label_file ||
        !pred_file)
    {
        printf("\nERROR: Could not open one or more reference files.\n");

        if (raw_file) fclose(raw_file);
        if (feat_file) fclose(feat_file);
        if (prob_file) fclose(prob_file);
        if (label_file) fclose(label_file);
        if (pred_file) fclose(pred_file);

        return 1;
    }

    const char *feature_names[FEATURE_COUNT] = {
        "crest",
        "skew",
        "kurt",
        "d1_ratio",
        "d2_ratio",
        "peak_asym",
        "energy_asym",
        "frac_small"
    };

    float max_feature_diff[FEATURE_COUNT] = {0.0f};
    int feature_pass_count[FEATURE_COUNT] = {0};

    int full_feature_pass = 0;
    int prediction_pass = 0;

    int prediction_mismatches = 0;
    int probability_mismatches = 0;

    float max_probability_diff = 0.0f;

    int tn = 0;
    int fp = 0;
    int fn = 0;
    int tp = 0;

    float raw_samples[CYCLE_LEN];
    float python_features[FEATURE_COUNT];

    for (int record = 0; record < TEST_RECORDS; record++) {

        /* Read raw 1000-sample waveform */
        size_t raw_read = fread(
            raw_samples,
            sizeof(float),
            CYCLE_LEN,
            raw_file
        );

        if (raw_read != CYCLE_LEN) {
            printf(
                "\nERROR: Raw sample read failed at "
                "record %d. Read %zu / %d samples.\n",
                record,
                raw_read,
                CYCLE_LEN
            );
            return 1;
        }

        /* Read Python feature vector */
        size_t feat_read = fread(
            python_features,
            sizeof(float),
            FEATURE_COUNT,
            feat_file
        );

        if (feat_read != FEATURE_COUNT) {
            printf(
                "\nERROR: Feature read failed at record %d.\n",
                record
            );
            return 1;
        }

        /* Read Python probability */
        float python_prob = 0.0f;

        if (fread(
                &python_prob,
                sizeof(float),
                1,
                prob_file
            ) != 1)
        {
            printf(
                "\nERROR: Probability read failed "
                "at record %d.\n",
                record
            );
            return 1;
        }

        /* Read true binary label */
        uint8_t python_label = 0;

        if (fread(
                &python_label,
                sizeof(uint8_t),
                1,
                label_file
            ) != 1)
        {
            printf(
                "\nERROR: Label read failed "
                "at record %d.\n",
                record
            );
            return 1;
        }

        /* Read Python prediction */
        uint8_t python_prediction = 0;

        if (fread(
                &python_prediction,
                sizeof(uint8_t),
                1,
                pred_file
            ) != 1)
        {
            printf(
                "\nERROR: Prediction read failed "
                "at record %d.\n",
                record
            );
            return 1;
        }

        /* C feature extraction */
        float c_features[FEATURE_COUNT];

        extract_features_8(
            raw_samples,
            c_features
        );

        int all_features_ok = 1;

        /* Compare all 8 features */
        for (int f = 0; f < FEATURE_COUNT; f++) {

            float diff = fabsf(
                c_features[f] - python_features[f]
            );

            if (diff > max_feature_diff[f]) {
                max_feature_diff[f] = diff;
            }

            if (diff <= FEATURE_TOLERANCE) {
                feature_pass_count[f]++;
            }
            else {
                all_features_ok = 0;
            }
        }

        if (all_features_ok) {
            full_feature_pass++;
        }

        /* C MLP inference */
        float c_probability = 0.0f;

        predict_mlp_c(
            c_features,
            &c_probability
        );

        int c_prediction =
            (c_probability >= 0.5f) ? 1 : 0;

        /* Prediction comparison */
        if (c_prediction == (int)python_prediction) {
            prediction_pass++;
        }
        else {
            prediction_mismatches++;

            if (prediction_mismatches <= 10) {
                printf(
                    "\n[PREDICTION MISMATCH] Record %d\n",
                    record
                );

                printf(
                    "  Python probability : %.8f\n",
                    python_prob
                );

                printf(
                    "  C probability      : %.8f\n",
                    c_probability
                );

                printf(
                    "  Python prediction  : %d\n",
                    python_prediction
                );

                printf(
                    "  C prediction       : %d\n",
                    c_prediction
                );
            }
        }

        /* Probability comparison */
        float probability_diff = fabsf(
            c_probability - python_prob
        );

        if (probability_diff > max_probability_diff) {
            max_probability_diff = probability_diff;
        }

        if (probability_diff > PROB_TOLERANCE) {
            probability_mismatches++;
        }

        /* C confusion matrix */
        if (python_label == 0 && c_prediction == 0) {
            tn++;
        }
        else if (python_label == 0 && c_prediction == 1) {
            fp++;
        }
        else if (python_label == 1 && c_prediction == 0) {
            fn++;
        }
        else if (python_label == 1 && c_prediction == 1) {
            tp++;
        }
    }

    fclose(raw_file);
    fclose(feat_file);
    fclose(prob_file);
    fclose(label_file);
    fclose(pred_file);

    printf(
        "\n%-12s | %-18s | %-18s\n",
        "Feature",
        "Max Abs Diff",
        "Pass Rate (<=1e-2)"
    );

    printf(
        "------------------------------------------------------\n"
    );

    for (int f = 0; f < FEATURE_COUNT; f++) {
        printf(
            "%-12s | %18.8e | %d / %d\n",
            feature_names[f],
            max_feature_diff[f],
            feature_pass_count[f],
            TEST_RECORDS
        );
    }

    int total = tn + fp + fn + tp;
    int normal_total = tn + fp;
    int arc_total = fn + tp;

    float accuracy =
        total > 0
        ? 100.0f * (float)(tn + tp) / (float)total
        : 0.0f;

    float false_trip_rate =
        normal_total > 0
        ? 100.0f * (float)fp / (float)normal_total
        : 0.0f;

    float arc_detection =
        arc_total > 0
        ? 100.0f * (float)tp / (float)arc_total
        : 0.0f;

    printf(
        "\n====================================================\n"
    );

    printf(
        "FULL TEST FEATURE EQUIVALENCE : %d / %d\n",
        full_feature_pass,
        TEST_RECORDS
    );

    printf(
        "FULL TEST PREDICTION MATCH    : %d / %d\n",
        prediction_pass,
        TEST_RECORDS
    );

    printf(
        "Prediction mismatches         : %d\n",
        prediction_mismatches
    );

    printf(
        "Probability mismatches >1e-4 : %d\n",
        probability_mismatches
    );

    printf(
        "Maximum probability difference: %.10e\n",
        max_probability_diff
    );

    printf("\nC CONFUSION MATRIX\n");

    printf("TN = %d\n", tn);
    printf("FP = %d\n", fp);
    printf("FN = %d\n", fn);
    printf("TP = %d\n", tp);

    printf(
        "\nC ACCURACY           : %.4f%%\n",
        accuracy
    );

    printf(
        "C FALSE-TRIP RATE    : %.4f%%\n",
        false_trip_rate
    );

    printf(
        "C ARC DETECTION      : %.4f%%\n",
        arc_detection
    );

    printf(
        "====================================================\n"
    );

    if (full_feature_pass == TEST_RECORDS &&
        prediction_pass == TEST_RECORDS)
    {
        printf(
            "\nPASS: C and Python pipelines are equivalent "
            "on all %d official test windows.\n",
            TEST_RECORDS
        );
    }
    else {
        printf(
            "\nFAIL: C/Python equivalence is incomplete.\n"
        );
    }

    return 0;
}