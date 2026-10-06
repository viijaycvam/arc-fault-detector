#include <Arduino.h>

extern "C" {
#include "feature_extraction.h"
}

#include "tinyml_engine_b.h"
#include "golden_cycles.h"

#define FEATURE_COUNT 8
#define FEATURE_TOLERANCE 1e-2f

static const char *feature_names[FEATURE_COUNT] = {
    "crest",
    "skew",
    "kurt",
    "d1_ratio",
    "d2_ratio",
    "peak_asym",
    "energy_asym",
    "frac_small"
};


static float max_feature_diff[FEATURE_COUNT] = {
    0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f
};

static int feature_pass_count[FEATURE_COUNT] = {
    0, 0, 0, 0,
    0, 0, 0, 0
};


void setup()
{
    Serial.begin(115200);

    delay(1500);

    Serial.println();
    Serial.println("====================================================");
    Serial.println("ESP32 100-CYCLE GOLDEN VERIFICATION");
    Serial.println("====================================================");

    Serial.println("Board : DOIT ESP32 DEVKIT V1");
    Serial.println("Port  : COM3");

    Serial.printf(
        "Golden cycles : %d\n",
        NUM_GOLDEN_CYCLES
    );

    Serial.printf(
        "Samples/cycle : %d\n",
        CYCLE_SAMPLES
    );

    Serial.println();

    int feature_cycle_pass = 0;
    int prediction_pass = 0;
    int probability_mismatch_count = 0;

    int first_mismatch_count = 0;

    uint64_t dsp_total_us = 0;
    uint64_t mlp_total_us = 0;
    uint64_t total_total_us = 0;

    uint32_t dsp_min_us = UINT32_MAX;
    uint32_t mlp_min_us = UINT32_MAX;
    uint32_t total_min_us = UINT32_MAX;

    uint32_t dsp_max_us = 0;
    uint32_t mlp_max_us = 0;
    uint32_t total_max_us = 0;

    float max_probability_diff = 0.0f;

    /*
     * Process every golden cycle.
     */
    for (int cycle = 0; cycle < NUM_GOLDEN_CYCLES; cycle++) {

        const GoldenCycle *gc =
            &GOLDEN_CYCLES[cycle];

        float computed_features[FEATURE_COUNT];

        /*
         * ----------------------------------------------------
         * DSP timing
         * ----------------------------------------------------
         */
        uint32_t dsp_start = micros();

        extract_features_8(
            gc->raw_samples,
            computed_features
        );

        uint32_t dsp_end = micros();

        /*
         * ----------------------------------------------------
         * MLP timing
         * ----------------------------------------------------
         */
        float probability = 0.0f;

        uint32_t mlp_start = micros();

        int prediction =
            predict_arc_8(
                computed_features,
                &probability
            );

        uint32_t mlp_end = micros();

        uint32_t dsp_time =
            dsp_end - dsp_start;

        uint32_t mlp_time =
            mlp_end - mlp_start;

        uint32_t total_time =
            mlp_end - dsp_start;

        /*
         * Accumulate timing.
         */
        dsp_total_us += dsp_time;
        mlp_total_us += mlp_time;
        total_total_us += total_time;

        if (dsp_time < dsp_min_us)
            dsp_min_us = dsp_time;

        if (mlp_time < mlp_min_us)
            mlp_min_us = mlp_time;

        if (total_time < total_min_us)
            total_min_us = total_time;

        if (dsp_time > dsp_max_us)
            dsp_max_us = dsp_time;

        if (mlp_time > mlp_max_us)
            mlp_max_us = mlp_time;

        if (total_time > total_max_us)
            total_max_us = total_time;


        /*
         * ----------------------------------------------------
         * Feature comparison
         * ----------------------------------------------------
         */
        bool all_features_ok = true;

        for (int f = 0; f < FEATURE_COUNT; f++) {

            float diff = fabsf(
                computed_features[f]
                - gc->expected_features[f]
            );

            if (diff > max_feature_diff[f]) {
                max_feature_diff[f] = diff;
            }

            if (diff <= FEATURE_TOLERANCE) {
                feature_pass_count[f]++;
            }
            else {
                all_features_ok = false;
            }
        }

        if (all_features_ok) {
            feature_cycle_pass++;
        }


        /*
         * ----------------------------------------------------
         * Prediction comparison
         * ----------------------------------------------------
         */
        int expected_prediction =
            (gc->expected_prob >= 0.5f)
            ? 1
            : 0;

        if (prediction == expected_prediction) {
            prediction_pass++;
        }
        else {

            if (first_mismatch_count < 10) {

                Serial.println();
                Serial.println(
                    "[PREDICTION MISMATCH]"
                );

                Serial.printf(
                    "Cycle          : %d\n",
                    cycle
                );

                Serial.printf(
                    "Record ID      : %d\n",
                    gc->record_id
                );

                Serial.printf(
                    "Expected label : %d\n",
                    gc->expected_label
                );

                Serial.printf(
                    "Expected prob  : %.8f\n",
                    gc->expected_prob
                );

                Serial.printf(
                    "ESP32 prob     : %.8f\n",
                    probability
                );

                Serial.printf(
                    "Expected pred  : %d\n",
                    expected_prediction
                );

                Serial.printf(
                    "ESP32 pred     : %d\n",
                    prediction
                );

                first_mismatch_count++;
            }
        }


        /*
         * ----------------------------------------------------
         * Probability comparison
         * ----------------------------------------------------
         */
        float probability_diff =
            fabsf(
                probability
                - gc->expected_prob
            );

        if (probability_diff > max_probability_diff) {
            max_probability_diff =
                probability_diff;
        }

        /*
         * Probability tolerance is informational here.
         * Prediction parity is the primary requirement.
         */
        if (probability_diff > 1e-4f) {
            probability_mismatch_count++;
        }
    }


    /*
     * --------------------------------------------------------
     * Calculate averages
     * --------------------------------------------------------
     */
    float avg_dsp_us =
        (float)dsp_total_us /
        (float)NUM_GOLDEN_CYCLES;

    float avg_mlp_us =
        (float)mlp_total_us /
        (float)NUM_GOLDEN_CYCLES;

    float avg_total_us =
        (float)total_total_us /
        (float)NUM_GOLDEN_CYCLES;


    /*
     * --------------------------------------------------------
     * Feature results
     * --------------------------------------------------------
     */
    Serial.println();
    Serial.println(
        "FEATURE EQUIVALENCE"
    );

    Serial.println(
        "----------------------------------------------------"
    );

    Serial.printf(
        "%-12s | %-18s | %-18s\n",
        "Feature",
        "Max Abs Diff",
        "Pass Rate"
    );

    Serial.println(
        "----------------------------------------------------"
    );

    for (int f = 0; f < FEATURE_COUNT; f++) {

        Serial.printf(
            "%-12s | %18.8e | %d / %d\n",
            feature_names[f],
            max_feature_diff[f],
            feature_pass_count[f],
            NUM_GOLDEN_CYCLES
        );
    }


    /*
     * --------------------------------------------------------
     * Prediction + timing results
     * --------------------------------------------------------
     */
    Serial.println();
    Serial.println(
        "===================================================="
    );

    Serial.printf(
        "DSP Feature Match Rate  : %d / %d\n",
        feature_cycle_pass,
        NUM_GOLDEN_CYCLES
    );

    Serial.printf(
        "MLP Prediction Match    : %d / %d\n",
        prediction_pass,
        NUM_GOLDEN_CYCLES
    );

    Serial.printf(
        "Probability diff >1e-4  : %d / %d\n",
        probability_mismatch_count,
        NUM_GOLDEN_CYCLES
    );

    Serial.printf(
        "Maximum probability diff: %.8e\n",
        max_probability_diff
    );

    Serial.println(
        "----------------------------------------------------"
    );

    Serial.printf(
        "Avg DSP latency   : %.2f us\n",
        avg_dsp_us
    );

    Serial.printf(
        "Avg MLP latency   : %.2f us\n",
        avg_mlp_us
    );

    Serial.printf(
        "Avg Total latency : %.2f us\n",
        avg_total_us
    );

    Serial.println();

    Serial.printf(
        "Min DSP latency   : %lu us\n",
        (unsigned long)dsp_min_us
    );

    Serial.printf(
        "Max DSP latency   : %lu us\n",
        (unsigned long)dsp_max_us
    );

    Serial.printf(
        "Min MLP latency   : %lu us\n",
        (unsigned long)mlp_min_us
    );

    Serial.printf(
        "Max MLP latency   : %lu us\n",
        (unsigned long)mlp_max_us
    );

    Serial.printf(
        "Min Total latency : %lu us\n",
        (unsigned long)total_min_us
    );

    Serial.printf(
        "Max Total latency : %lu us\n",
        (unsigned long)total_max_us
    );

    Serial.println(
        "===================================================="
    );


    /*
     * --------------------------------------------------------
     * Final PASS/FAIL
     * --------------------------------------------------------
     */
    if (
        feature_cycle_pass == NUM_GOLDEN_CYCLES &&
        prediction_pass == NUM_GOLDEN_CYCLES
    ) {

        Serial.println();
        Serial.println(
            "STATUS: ESP32 GOLDEN VERIFICATION PASSED"
        );

        Serial.println(
            "100/100 feature and prediction matches."
        );

    }
    else {

        Serial.println();
        Serial.println(
            "STATUS: ESP32 GOLDEN VERIFICATION FAILED"
        );

        Serial.printf(
            "Feature cycles passed: %d / %d\n",
            feature_cycle_pass,
            NUM_GOLDEN_CYCLES
        );

        Serial.printf(
            "Predictions passed: %d / %d\n",
            prediction_pass,
            NUM_GOLDEN_CYCLES
        );
    }

    Serial.println();
}


void loop()
{
    /*
     * Test runs once in setup().
     */
    delay(10000);
}