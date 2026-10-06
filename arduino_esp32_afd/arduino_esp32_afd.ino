#include <Arduino.h>
#include <math.h>

extern "C" {
#include "feature_extraction.h"
}

#include "tinyml_engine_b.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SAMPLE_COUNT 1000
#define SAMPLE_RATE_HZ 50000.0f

static float test_samples[SAMPLE_COUNT];


/*
 * Create a safe software-only waveform.
 *
 * This is NOT a real arc waveform.
 * It is only used to verify that the ESP32
 * can execute the same DSP + MLP code.
 */
static void generate_test_waveform()
{
    for (int i = 0; i < SAMPLE_COUNT; i++) {

        float t =
            (float)i / SAMPLE_RATE_HZ;

        float fundamental =
            0.8f *
            sinf(
                2.0f * (float)M_PI *
                50.0f *
                t
            );

        float harmonic =
            0.08f *
            sinf(
                2.0f * (float)M_PI *
                150.0f *
                t
            );

        /*
         * DC offset keeps the synthetic waveform
         * positive, similar to an ADC-biased signal.
         */
        test_samples[i] =
            1.65f +
            fundamental +
            harmonic;
    }
}


void setup()
{
    Serial.begin(115200);

    delay(1000);

    Serial.println();
    Serial.println("==========================================");
    Serial.println("ARC-FAULT DETECTOR - ESP32 SOFTWARE TEST");
    Serial.println("==========================================");

    Serial.println("Board : DOIT ESP32 DEVKIT V1");
    Serial.println("Port  : COM3");
    Serial.println();

    Serial.println("Generating 1000-sample waveform...");

    generate_test_waveform();


    /*
     * ---------------------------------------------------------
     * DSP timing
     * ---------------------------------------------------------
     */
    float features[8];

    uint32_t dsp_start = micros();

    extract_features_8(
        test_samples,
        features
    );

    uint32_t dsp_end = micros();


    /*
     * ---------------------------------------------------------
     * MLP timing
     * ---------------------------------------------------------
     */
    float probability = 0.0f;

    uint32_t mlp_start = micros();

    int prediction =
        predict_arc_8(
            features,
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
     * ---------------------------------------------------------
     * Display features
     * ---------------------------------------------------------
     */
    Serial.println();
    Serial.println("FEATURE VECTOR");
    Serial.println("------------------------------------------");

    Serial.printf(
        "0 crest       = %.8f\n",
        features[0]
    );

    Serial.printf(
        "1 skew        = %.8f\n",
        features[1]
    );

    Serial.printf(
        "2 kurt        = %.8f\n",
        features[2]
    );

    Serial.printf(
        "3 d1_ratio    = %.8f\n",
        features[3]
    );

    Serial.printf(
        "4 d2_ratio    = %.8f\n",
        features[4]
    );

    Serial.printf(
        "5 peak_asym   = %.8f\n",
        features[5]
    );

    Serial.printf(
        "6 energy_asym = %.8f\n",
        features[6]
    );

    Serial.printf(
        "7 frac_small  = %.8f\n",
        features[7]
    );


    /*
     * ---------------------------------------------------------
     * Display inference
     * ---------------------------------------------------------
     */
    Serial.println();
    Serial.println("MLP INFERENCE");
    Serial.println("------------------------------------------");

    Serial.printf(
        "Arc probability = %.8f\n",
        probability
    );

    Serial.printf(
        "Prediction      = %s\n",
        prediction ? "ARC" : "NORMAL"
    );


    /*
     * ---------------------------------------------------------
     * Display timing
     * ---------------------------------------------------------
     */
    Serial.println();
    Serial.println("TIMING");
    Serial.println("------------------------------------------");

    Serial.printf(
        "DSP time        = %lu us\n",
        (unsigned long)dsp_time
    );

    Serial.printf(
        "MLP time        = %lu us\n",
        (unsigned long)mlp_time
    );

    Serial.printf(
        "Total time      = %lu us\n",
        (unsigned long)total_time
    );

    Serial.printf(
        "Total time      = %.3f ms\n",
        total_time / 1000.0f
    );


    /*
     * ---------------------------------------------------------
     * Final status
     * ---------------------------------------------------------
     */
    Serial.println();
    Serial.println("==========================================");
    Serial.println("ESP32 AFD SOFTWARE TEST COMPLETE");
    Serial.println("==========================================");
}


void loop()
{
    delay(5000);
}