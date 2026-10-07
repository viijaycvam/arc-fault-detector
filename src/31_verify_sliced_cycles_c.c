#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "feature_extraction.h"

#define FEATURE_COUNT 8
#define MAX_RECORDS 1000
#define FEATURE_TOLERANCE 1e-2f

int main(void)
{
    const char *path =
        "reports/sliced_cycles_golden.bin";

    FILE *fp = fopen(path, "rb");

    if (!fp) {
        perror("fopen");
        return 1;
    }

    char magic[4];

    uint32_t version = 0;
    uint32_t records = 0;
    uint32_t cycle_len = 0;
    uint32_t feature_count = 0;

    /* --------------------------------------------------------
       Read file header
       -------------------------------------------------------- */

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

    if (memcmp(magic, "AFSG", 4) != 0) {
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

    float max_diff[FEATURE_COUNT] = {
        0.0f
    };

    int feature_pass[FEATURE_COUNT] = {
        0
    };

    int all_cycle_pass = 0;

    /* --------------------------------------------------------
       Read and verify every sliced cycle
       -------------------------------------------------------- */

    for (
        uint32_t record = 0;
        record < records;
        record++
    ) {
        float raw[CYCLE_LEN];

        float expected_features[
            FEATURE_COUNT
        ];

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

        /* Existing verified C implementation */
        extract_features_8(
            raw,
            computed_features
        );

        int cycle_ok = 1;

        for (
            int f = 0;
            f < FEATURE_COUNT;
            f++
        ) {
            float diff = fabsf(
                computed_features[f] -
                expected_features[f]
            );

            if (diff > max_diff[f]) {
                max_diff[f] = diff;
            }

            if (diff <= FEATURE_TOLERANCE) {
                feature_pass[f]++;
            }
            else {
                cycle_ok = 0;
            }
        }

        if (cycle_ok) {
            all_cycle_pass++;
        }
    }

    /* --------------------------------------------------------
       Check for unexpected trailing data
       -------------------------------------------------------- */

    int no_trailing_data =
        (fgetc(fp) == EOF);

    fclose(fp);

    /* --------------------------------------------------------
       Print result
       -------------------------------------------------------- */

    printf(
        "============================================================\n"
    );

    printf(
        " C VERIFICATION OF ROBUSTLY SLICED CYCLES\n"
    );

    printf(
        "============================================================\n\n"
    );

    printf(
        "Records                 : %u\n",
        records
    );

    printf(
        "Cycle length            : %u\n",
        cycle_len
    );

    printf(
        "Feature count           : %u\n",
        feature_count
    );

    printf(
        "Feature tolerance       : %.1e\n",
        FEATURE_TOLERANCE
    );

    printf(
        "All-8 feature match     : %d / %u\n",
        all_cycle_pass,
        records
    );

    printf(
        "Binary file fully read  : %s\n",
        no_trailing_data
            ? "YES"
            : "NO"
    );

    printf(
        "\n%-14s | %-18s | %-12s\n",
        "Feature",
        "Max Abs Diff",
        "Pass"
    );

    printf(
        "------------------------------------------------------------\n"
    );

    for (
        int f = 0;
        f < FEATURE_COUNT;
        f++
    ) {
        printf(
            "%-14s | %18.8e | %d / %u\n",
            feature_names[f],
            max_diff[f],
            feature_pass[f],
            records
        );
    }

    printf(
        "\n------------------------------------------------------------\n"
    );

    int pass =
        (
            all_cycle_pass ==
            (int)records
        ) &&
        no_trailing_data;

    if (pass) {
        printf(
            "STATUS: C SLICED-CYCLE FEATURE VERIFICATION PASSED\n"
        );
    }
    else {
        printf(
            "STATUS: C SLICED-CYCLE FEATURE VERIFICATION FAILED\n"
        );
    }

    printf(
        "------------------------------------------------------------\n"
    );

    return pass ? 0 : 1;
}