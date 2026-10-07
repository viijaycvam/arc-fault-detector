#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "feature_extraction.h"
#include "model_weights_b.h"

#define FEATURE_COUNT 8
#define MAX_RECORDS 1000

#define FEATURE_TOLERANCE 1e-2f
#define PROB_TOLERANCE 1e-4f

static inline float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static float predict_mlp_c(
    const float features[FEATURE_COUNT]
)
{
    float scaled[FEATURE_COUNT];
    float h1[16];
    float h2[8];

    float out_raw = B3_B[0];

    for (int i = 0; i < FEATURE_COUNT; i++) {
        scaled[i] =
            (
                features[i] -
                SCALER_MEAN_B[i]
            ) /
            SCALER_SCALE_B[i];
    }

    for (int j = 0; j < 16; j++) {

        float sum = B1_B[j];

        for (int i = 0; i < FEATURE_COUNT; i++) {
            sum +=
                scaled[i] *
                W1_B[i][j];
        }

        h1[j] = relu(sum);
    }

    for (int j = 0; j < 8; j++) {

        float sum = B2_B[j];

        for (int i = 0; i < 16; i++) {
            sum +=
                h1[i] *
                W2_B[i][j];
        }

        h2[j] = relu(sum);
    }

    for (int i = 0; i < 8; i++) {
        out_raw +=
            h2[i] *
            W3_B[i][0];
    }

    return sigmoid(out_raw);
}

int main(void)
{
    const char *path =
        "reports/sliced_full_golden.bin";

    FILE *fp = fopen(
        path,
        "rb"
    );

    if (!fp) {
        perror("fopen");
        return 1;
    }

    char magic[4];

    uint32_t version = 0;
    uint32_t records = 0;
    uint32_t cycle_len = 0;
    uint32_t feature_count = 0;

    if (
        fread(magic, 1, 4, fp) != 4 ||
        fread(&version, sizeof(version), 1, fp) != 1 ||
        fread(&records, sizeof(records), 1, fp) != 1 ||
        fread(&cycle_len, sizeof(cycle_len), 1, fp) != 1 ||
        fread(&feature_count, sizeof(feature_count), 1, fp) != 1
    ) {
        fprintf(
            stderr,
            "Failed to read file header.\n"
        );

        fclose(fp);
        return 1;
    }

    if (memcmp(magic, "AFGF", 4) != 0) {
        fprintf(
            stderr,
            "Invalid file magic.\n"
        );

        fclose(fp);
        return 1;
    }

    if (
        version != 1 ||
        cycle_len != CYCLE_LEN ||
        feature_count != FEATURE_COUNT
    ) {
        fprintf(
            stderr,
            "Header mismatch:\n"
            "  version=%u\n"
            "  cycle_len=%u\n"
            "  features=%u\n",
            version,
            cycle_len,
            feature_count
        );

        fclose(fp);
        return 1;
    }

    if (
        records == 0 ||
        records > MAX_RECORDS
    ) {
        fprintf(
            stderr,
            "Invalid record count: %u\n",
            records
        );

        fclose(fp);
        return 1;
    }

    float max_feature_diff[FEATURE_COUNT] = {
        0.0f
    };

    float max_prob_diff = 0.0f;

    int feature_cycles_pass = 0;
    int probability_pass = 0;
    int prediction_pass = 0;

    for (
        uint32_t record = 0;
        record < records;
        record++
    ) {
        float raw[CYCLE_LEN];

        float expected_features[
            FEATURE_COUNT
        ];

        float expected_prob = 0.0f;

        float computed_features[
            FEATURE_COUNT
        ];

        if (
            fread(
                raw,
                sizeof(float),
                CYCLE_LEN,
                fp
            ) != CYCLE_LEN
        ) {
            fprintf(
                stderr,
                "Short raw-cycle read at record %u.\n",
                record
            );

            fclose(fp);
            return 1;
        }

        if (
            fread(
                expected_features,
                sizeof(float),
                FEATURE_COUNT,
                fp
            ) != FEATURE_COUNT
        ) {
            fprintf(
                stderr,
                "Short feature read at record %u.\n",
                record
            );

            fclose(fp);
            return 1;
        }

        if (
            fread(
                &expected_prob,
                sizeof(float),
                1,
                fp
            ) != 1
        ) {
            fprintf(
                stderr,
                "Short probability read at record %u.\n",
                record
            );

            fclose(fp);
            return 1;
        }

        extract_features_8(
            raw,
            computed_features
        );

        int feature_ok = 1;

        for (
            int f = 0;
            f < FEATURE_COUNT;
            f++
        ) {
            float diff = fabsf(
                computed_features[f] -
                expected_features[f]
            );

            if (diff > max_feature_diff[f]) {
                max_feature_diff[f] = diff;
            }

            if (diff > FEATURE_TOLERANCE) {
                feature_ok = 0;
            }
        }

        if (feature_ok) {
            feature_cycles_pass++;
        }

        float prob_c =
            predict_mlp_c(
                computed_features
            );

        float prob_diff =
            fabsf(
                prob_c -
                expected_prob
            );

        if (prob_diff > max_prob_diff) {
            max_prob_diff = prob_diff;
        }

        if (prob_diff <= PROB_TOLERANCE) {
            probability_pass++;
        }

        int pred_c =
            (prob_c >= 0.5f)
            ? 1
            : 0;

        int pred_expected =
            (expected_prob >= 0.5f)
            ? 1
            : 0;

        if (pred_c == pred_expected) {
            prediction_pass++;
        }

        printf(
            "Record %2u | "
            "Py Prob: %.8f | "
            "C Prob: %.8f | "
            "Diff: %.3e | "
            "Pred: %s\n",
            record,
            expected_prob,
            prob_c,
            prob_diff,
            pred_c ? "ARC" : "NORMAL"
        );
    }

    int no_trailing_data =
        (fgetc(fp) == EOF);

    fclose(fp);

    printf(
        "\n============================================================\n"
    );

    printf(
        " C VERIFICATION OF COMPLETE SLICED-CYCLE PIPELINE\n"
    );

    printf(
        "============================================================\n\n"
    );

    printf(
        "Records                         : %u\n",
        records
    );

    printf(
        "Cycle length                    : %u\n",
        cycle_len
    );

    printf(
        "Feature tolerance               : %.1e\n",
        FEATURE_TOLERANCE
    );

    printf(
        "Probability tolerance           : %.1e\n",
        PROB_TOLERANCE
    );

    printf(
        "All-8 feature match             : %d / %u\n",
        feature_cycles_pass,
        records
    );

    printf(
        "MLP probability match           : %d / %u\n",
        probability_pass,
        records
    );

    printf(
        "Binary prediction match         : %d / %u\n",
        prediction_pass,
        records
    );

    printf(
        "Max probability absolute diff   : %.8e\n",
        max_prob_diff
    );

    printf(
        "Binary file fully read          : %s\n",
        no_trailing_data
            ? "YES"
            : "NO"
    );

    printf(
        "\nFeature maximum absolute errors:\n"
    );

    const char *names[FEATURE_COUNT] = {
        "crest",
        "skew",
        "kurt",
        "d1_ratio",
        "d2_ratio",
        "peak_asym",
        "energy_asym",
        "frac_small"
    };

    for (
        int f = 0;
        f < FEATURE_COUNT;
        f++
    ) {
        printf(
            "  %-12s : %.8e\n",
            names[f],
            max_feature_diff[f]
        );
    }

    int pass =
        (
            feature_cycles_pass ==
            (int)records
        ) &&
        (
            probability_pass ==
            (int)records
        ) &&
        (
            prediction_pass ==
            (int)records
        ) &&
        no_trailing_data;

    printf(
        "\n------------------------------------------------------------\n"
    );

    if (pass) {
        printf(
            "STATUS: COMPLETE SLICED-CYCLE C/MLP VERIFICATION PASSED\n"
        );
    }
    else {
        printf(
            "STATUS: COMPLETE SLICED-CYCLE C/MLP VERIFICATION FAILED\n"
        );
    }

    printf(
        "------------------------------------------------------------\n"
    );

    return pass ? 0 : 1;
}