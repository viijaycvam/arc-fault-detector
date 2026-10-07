#include <Arduino.h>
#include <math.h>

extern "C" {
#include "feature_extraction.h"
}

#include "tinyml_engine_b.h"
#include "golden_cycles.h"


// ============================================================
// ESP32 DATASET-GOLDEN 3-OF-5 TEMPORAL VOTING TEST
//
// NO ADC
// NO DAC
// NO WIRING REQUIRED
//
// Uses the actual raw 1000-sample dataset cycles stored in:
//
//     golden_cycles.h
//
// Test sequence:
//
//     NORMAL
//     NORMAL
//     ARC
//     ARC
//     ARC
//
// The ESP32 independently performs:
//
//     raw cycle
//       -> 8 features
//       -> scaler + MLP
//       -> probability
//       -> binary prediction
//
// Then:
//
//     3-of-5 temporal vote
//
// IMPORTANT:
// This validates embedded inference + voting using real dataset
// cycles already exported into the project.
//
// It does NOT measure ADC performance.
// It does NOT represent field performance.
// ============================================================


#define VOTE_CYCLES             5

#define ARC_VOTE_THRESHOLD      3

#define PROBABILITY_THRESHOLD   0.5f

#define FEATURE_TOLERANCE       1e-2f

#define PROBABILITY_TOLERANCE   1e-4f


// Desired test pattern.
//
// 0 = NORMAL
// 1 = ARC
//
static const int DESIRED_PATTERN[VOTE_CYCLES] =
{
    0,
    0,
    1,
    1,
    1
};


static float cycle_probabilities[
    VOTE_CYCLES
];

static int cycle_predictions[
    VOTE_CYCLES
];

static int selected_record_ids[
    VOTE_CYCLES
];


// ------------------------------------------------------------
// Find dataset golden cycles matching desired labels.
//
// golden_cycles.h contains NUM_GOLDEN_CYCLES entries with:
//   expected_label
//   expected_features
//   expected_prob
//   raw_samples[1000]
//
// We select the first unused entry for each required label.
// ------------------------------------------------------------

static bool select_cycles()
{
    bool used[
        NUM_GOLDEN_CYCLES
    ] = { false };


    for (
        int target = 0;
        target < VOTE_CYCLES;
        target++
    )
    {
        int wanted_label =
            DESIRED_PATTERN[target];


        bool found =
            false;


        for (
            int i = 0;
            i < NUM_GOLDEN_CYCLES;
            i++
        )
        {
            if (
                !used[i] &&
                GOLDEN_CYCLES[i].expected_label == wanted_label && ((wanted_label == 0 && GOLDEN_CYCLES[i].expected_prob < PROBABILITY_THRESHOLD) || (wanted_label == 1 && GOLDEN_CYCLES[i].expected_prob >= PROBABILITY_THRESHOLD))
            )
            {
                selected_record_ids[target] =
                    i;

                used[i] =
                    true;

                found =
                    true;

                break;
            }
        }


        if (!found)
        {
            return false;
        }
    }


    return true;
}


// ------------------------------------------------------------
// Print feature comparison
//
// The expected features are the Python reference values
// stored in golden_cycles.h.
//
// The computed features come from the actual ESP32 C extractor.
// ------------------------------------------------------------

static bool compare_features(
    const float computed[8],
    const float expected[8]
)
{
    bool pass =
        true;


    const char *names[8] =
    {
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
        int i = 0;
        i < 8;
        i++
    )
    {
        float diff =
            fabsf(
                computed[i] -
                expected[i]
            );


        if (
            diff >
            FEATURE_TOLERANCE
        )
        {
            pass =
                false;
        }


        Serial.printf(
            "    %-12s C=%.8f  Ref=%.8f  Diff=%.3e\n",
            names[i],
            computed[i],
            expected[i],
            diff
        );
    }


    return pass;
}


// ------------------------------------------------------------
// SETUP
// ------------------------------------------------------------

void setup()
{
    Serial.begin(
        115200
    );


    delay(
        1000
    );


    Serial.println();

    Serial.println(
        "===================================================="
    );

    Serial.println(
        "ESP32 DATASET-GOLDEN 3-OF-5 TEMPORAL VOTE TEST"
    );

    Serial.println(
        "===================================================="
    );


    Serial.printf(
        "Golden cycles available : %d\n",
        NUM_GOLDEN_CYCLES
    );


    Serial.printf(
        "Vote policy             : %d-of-%d\n",
        ARC_VOTE_THRESHOLD,
        VOTE_CYCLES
    );


    Serial.printf(
        "Probability threshold   : %.3f\n",
        PROBABILITY_THRESHOLD
    );


    Serial.println();

    Serial.println(
        "D25/D34 wiring is NOT used."
    );


    Serial.println(
        "Using real dataset golden cycles."
    );


    // --------------------------------------------------------
    // Select 2 NORMAL + 3 ARC cycles.
    // --------------------------------------------------------

    if (
        !select_cycles()
    )
    {
        Serial.println();

        Serial.println(
            "STATUS: GOLDEN CYCLE SELECTION FAILED"
        );


        while (true)
        {
            delay(
                1000
            );
        }
    }


    Serial.println();

    Serial.println(
        "SELECTED DATASET SEQUENCE"
    );

    Serial.println(
        "---------------------------------------------"
    );


    for (
        int i = 0;
        i < VOTE_CYCLES;
        i++
    )
    {
        int golden_index =
            selected_record_ids[i];


        const GoldenCycle *gc =
            &GOLDEN_CYCLES[
                golden_index
            ];


        Serial.printf(
            "Cycle %d -> Golden index %d | "
            "Record ID %d | Expected %s | "
            "Reference prob %.8f\n",
            i + 1,
            golden_index,
            gc->record_id,
            gc->expected_label
                ? "ARC"
                : "NORMAL",
            gc->expected_prob
        );
    }


    // --------------------------------------------------------
    // Process five cycles.
    // --------------------------------------------------------

    int arc_votes =
        0;


    int feature_cycle_pass =
        0;


    int probability_pass =
        0;


    int prediction_pass =
        0;


    uint32_t total_processing_us =
        0;


    Serial.println();

    Serial.println(
        "PER-CYCLE INFERENCE"
    );

    Serial.println(
        "---------------------------------------------"
    );


    for (
        int cycle = 0;
        cycle < VOTE_CYCLES;
        cycle++
    )
    {
        int golden_index =
            selected_record_ids[
                cycle
            ];


        const GoldenCycle *gc =
            &GOLDEN_CYCLES[
                golden_index
            ];


        float features[8];


        // ----------------------------------------------------
        // C feature extraction
        // ----------------------------------------------------

        uint32_t cycle_start =
            micros();


        extract_features_8(
            gc->raw_samples,
            features
        );


        // ----------------------------------------------------
        // Compare features with stored Python reference.
        // ----------------------------------------------------

        bool feature_ok =
            compare_features(
                features,
                gc->expected_features
            );


        if (feature_ok)
        {
            feature_cycle_pass++;
        }


        // ----------------------------------------------------
        // C MLP
        // ----------------------------------------------------

        float probability =
            0.0f;


        int prediction =
            predict_arc_8(
                features,
                &probability
            );


        uint32_t cycle_end =
            micros();


        uint32_t processing_us =
            cycle_end -
            cycle_start;


        total_processing_us +=
            processing_us;


        cycle_probabilities[
            cycle
        ] =
            probability;


        cycle_predictions[
            cycle
        ] =
            prediction;


        // ----------------------------------------------------
        // Probability comparison.
        // ----------------------------------------------------

        float probability_diff =
            fabsf(
                probability -
                gc->expected_prob
            );


        bool probability_ok =
            (
                probability_diff <=
                PROBABILITY_TOLERANCE
            );


        if (probability_ok)
        {
            probability_pass++;
        }


        // ----------------------------------------------------
        // Binary prediction comparison.
        // ----------------------------------------------------

        bool prediction_ok =
            (
                prediction ==
                gc->expected_label
            );


        if (prediction_ok)
        {
            prediction_pass++;
        }


        // ----------------------------------------------------
        // Temporal vote.
        // ----------------------------------------------------

        if (
            probability >=
            PROBABILITY_THRESHOLD
        )
        {
            arc_votes++;
        }


        Serial.printf(
            "\nCycle %d\n",
            cycle + 1
        );


        Serial.printf(
            "  Record ID           : %d\n",
            gc->record_id
        );


        Serial.printf(
            "  Expected label      : %s\n",
            gc->expected_label
                ? "ARC"
                : "NORMAL"
        );


        Serial.printf(
            "  Computed probability: %.8f\n",
            probability
        );


        Serial.printf(
            "  Reference probability: %.8f\n",
            gc->expected_prob
        );


        Serial.printf(
            "  Probability diff    : %.8e\n",
            probability_diff
        );


        Serial.printf(
            "  Computed prediction : %s\n",
            prediction
                ? "ARC"
                : "NORMAL"
        );


        Serial.printf(
            "  Feature match       : %s\n",
            feature_ok
                ? "PASS"
                : "FAIL"
        );


        Serial.printf(
            "  Probability match   : %s\n",
            probability_ok
                ? "PASS"
                : "FAIL"
        );


        Serial.printf(
            "  Prediction match    : %s\n",
            prediction_ok
                ? "PASS"
                : "FAIL"
        );


        Serial.printf(
            "  Processing time     : %lu us\n",
            (unsigned long)
                processing_us
        );
    }


    // --------------------------------------------------------
    // Final 3-of-5 vote.
    // --------------------------------------------------------

    bool final_arc =
        (
            arc_votes >=
            ARC_VOTE_THRESHOLD
        );


    float mean_probability =
        0.0f;


    for (
        int i = 0;
        i < VOTE_CYCLES;
        i++
    )
    {
        mean_probability +=
            cycle_probabilities[i];
    }


    mean_probability /=
        (float)VOTE_CYCLES;


    // --------------------------------------------------------
    // Summary
    // --------------------------------------------------------

    Serial.println();

    Serial.println(
        "===================================================="
    );

    Serial.println(
        "VERIFICATION SUMMARY"
    );

    Serial.println(
        "===================================================="
    );


    Serial.printf(
        "Feature cycles matched : %d / %d\n",
        feature_cycle_pass,
        VOTE_CYCLES
    );


    Serial.printf(
        "Probability matches     : %d / %d\n",
        probability_pass,
        VOTE_CYCLES
    );


    Serial.printf(
        "Prediction matches      : %d / %d\n",
        prediction_pass,
        VOTE_CYCLES
    );


    Serial.printf(
        "Max vote probability    : not applicable\n"
    );


    Serial.printf(
        "ARC votes               : %d / %d\n",
        arc_votes,
        VOTE_CYCLES
    );


    Serial.printf(
        "Required ARC votes      : %d\n",
        ARC_VOTE_THRESHOLD
    );


    Serial.printf(
        "Mean probability        : %.8f\n",
        mean_probability
    );


    Serial.printf(
        "Total processing        : %lu us\n",
        (unsigned long)
            total_processing_us
    );


    Serial.printf(
        "Average cycle processing: %.1f us\n",
        (
            (float)
            total_processing_us
            /
            (float)VOTE_CYCLES
        )
    );


    Serial.printf(
        "Final decision          : %s\n",
        final_arc
            ? "ARC"
            : "NORMAL"
    );


    // --------------------------------------------------------
    // Overall pass condition.
    //
    // We require:
    //
    //   1. All 8 features match Python reference.
    //   2. All MLP probabilities match reference.
    //   3. All binary predictions match reference.
    //   4. The selected 2 NORMAL + 3 ARC sequence produces
    //      exactly the expected 3-of-5 ARC decision.
    // --------------------------------------------------------

    bool vote_ok =
        final_arc;


    bool overall_pass =
        (
            feature_cycle_pass ==
            VOTE_CYCLES
        )
        &&
        (
            probability_pass ==
            VOTE_CYCLES
        )
        &&
        (
            prediction_pass ==
            VOTE_CYCLES
        )
        &&
        vote_ok;


    Serial.println();

    Serial.println(
        "===================================================="
    );


    if (overall_pass)
    {
        Serial.println(
            "STATUS: DATASET 3-OF-5 VOTE TEST PASSED"
        );
    }
    else
    {
        Serial.println(
            "STATUS: DATASET 3-OF-5 VOTE TEST FAILED"
        );
    }


    Serial.println(
        "===================================================="
    );
}


void loop()
{
    delay(
        5000
    );
}