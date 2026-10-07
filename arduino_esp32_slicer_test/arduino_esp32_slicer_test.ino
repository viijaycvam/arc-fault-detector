#include <Arduino.h>
#include <math.h>

extern "C" {
#include "feature_extraction.h"
}

#include "tinyml_engine_b.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ============================================================
// ESP32 ROBUST CYCLE-SLICER ON-TARGET TEST
//
// SOFTWARE-GENERATED WAVEFORM ONLY
//
// No ADC input is used.
// D25 and D34 should remain DISCONNECTED.
//
// Sampling:
//   50 kHz
//
// Test frequencies:
//   49.5 Hz
//   50.0 Hz
//   50.5 Hz
//
// Waveform:
//   fundamental
//   + 5% deterministic noise
//   + 20% 3rd harmonic
//   + 10% 5th harmonic
//
// Processing:
//   raw waveform
//       |
//       +--> on-the-fly low-pass filter
//       |
//       +--> hysteresis crossing detector
//       |
//       +--> cycle boundaries
//       |
//       +--> resample RAW cycle to 1000 samples
//       |
//       +--> verified 8-feature extractor
//       |
//       +--> verified MLP
// ============================================================


// ------------------------------------------------------------
// Configuration
// ------------------------------------------------------------

#define FS_HZ                       50000.0f

#define MAX_SAMPLES                 21000
#define TARGET_CYCLE_LEN            1000
#define MAX_CROSSINGS               32

#define LOWPASS_CUTOFF_HZ          100.0f
#define HYSTERESIS                 0.10f
#define MIN_CROSSING_INTERVAL      700.0f

#define TEST_CYCLES                 20


// ------------------------------------------------------------
// Global buffers
//
// IMPORTANT:
// No filtered_samples[] array is used.
// The low-pass filter runs one sample at a time.
// This saves about 84 KB of DRAM.
// ------------------------------------------------------------

static float raw_samples[MAX_SAMPLES];

static float resampled_cycle[
    TARGET_CYCLE_LEN
];

static float crossing_positions[
    MAX_CROSSINGS
];


// ------------------------------------------------------------
// Deterministic pseudo-random generator
// ------------------------------------------------------------

static uint32_t noise_state =
    0x12345678UL;


static uint32_t next_noise_u32()
{
    uint32_t x =
        noise_state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;

    noise_state =
        x;

    return x;
}


// ------------------------------------------------------------
// Deterministic noise
//
// Produces approximately the requested RMS relative to the
// fundamental amplitude.
// ------------------------------------------------------------

static float deterministic_noise(
    float noise_rms
)
{
    uint32_t r =
        next_noise_u32();

    float u =
        (
            (float)(
                r & 0x00FFFFFFUL
            )
            /
            8388607.5f
        )
        -
        1.0f;

    return
        u *
        noise_rms /
        0.5773503f;
}


// ------------------------------------------------------------
// Generate test waveform
// ------------------------------------------------------------

static int generate_waveform(
    float frequency,
    float noise_rms,
    float h3,
    float h5
)
{
    float duration =
        (float)TEST_CYCLES /
        frequency;

    int sample_count =
        (int)ceilf(
            duration *
            FS_HZ
        );

    if (
        sample_count <= 0 ||
        sample_count > MAX_SAMPLES
    )
    {
        return 0;
    }

    noise_state =
        0x12345678UL;

    for (
        int i = 0;
        i < sample_count;
        i++
    )
    {
        float t =
            (float)i /
            FS_HZ;

        // Fundamental.
        float fundamental =
            sinf(
                2.0f *
                (float)M_PI *
                frequency *
                t
            );

        // 3rd harmonic.
        float third =
            h3 *
            sinf(
                2.0f *
                (float)M_PI *
                3.0f *
                frequency *
                t
            );

        // 5th harmonic.
        float fifth =
            h5 *
            sinf(
                2.0f *
                (float)M_PI *
                5.0f *
                frequency *
                t
            );

        // Deterministic noise.
        float noise =
            deterministic_noise(
                noise_rms
            );

        raw_samples[i] =
            fundamental +
            third +
            fifth +
            noise;
    }

    return sample_count;
}


// ------------------------------------------------------------
// Hysteresis + low-pass zero-crossing detector
//
// The low-pass filter is performed one sample at a time.
//
// This is intentionally equivalent to the Python algorithm:
//
//   filtered[n] = filtered[n-1]
//                 + alpha * (raw[n] - filtered[n-1])
//
// Then:
//
//   filtered <= -H
//       |
//       v
//     ARM
//       |
//       v
//   filtered crosses +H
//       |
//       v
//   accept crossing
//
// A minimum interval rejects impossible extra crossings.
// ------------------------------------------------------------

static int detect_crossings(
    const float *x,
    int count
)
{
    int crossing_count =
        0;

    int state =
        0;

    float last_crossing =
        -1.0f;


    // --------------------------------------------------------
    // First-order low-pass filter coefficient.
    // --------------------------------------------------------

    float dt =
        1.0f /
        FS_HZ;

    float rc =
        1.0f /
        (
            2.0f *
            (float)M_PI *
            LOWPASS_CUTOFF_HZ
        );

    float alpha =
        dt /
        (
            rc +
            dt
        );


    // --------------------------------------------------------
    // Initial filtered sample.
    // --------------------------------------------------------

    float previous_filtered =
        x[0];


    // --------------------------------------------------------
    // Process each sample.
    // --------------------------------------------------------

    for (
        int i = 1;
        i < count;
        i++
    )
    {
        float current_filtered =
            previous_filtered +
            alpha *
            (
                x[i] -
                previous_filtered
            );

        float previous =
            previous_filtered;

        float current =
            current_filtered;


        // ----------------------------------------------------
        // State 0:
        // Wait until signal goes below -H.
        // ----------------------------------------------------

        if (state == 0)
        {
            if (
                current <=
                -HYSTERESIS
            )
            {
                state =
                    1;
            }
        }


        // ----------------------------------------------------
        // State 1:
        // Wait for rising transition through +H.
        // ----------------------------------------------------

        else
        {
            if (
                previous <
                HYSTERESIS
                &&
                current >=
                HYSTERESIS
            )
            {
                float denominator =
                    current -
                    previous;

                float crossing;


                // --------------------------------------------
                // Linear interpolation.
                // --------------------------------------------

                if (
                    fabsf(
                        denominator
                    ) <
                    1e-15f
                )
                {
                    crossing =
                        (float)i;
                }
                else
                {
                    crossing =
                        (float)(i - 1)
                        +
                        (
                            (
                                HYSTERESIS -
                                previous
                            )
                            /
                            denominator
                        );
                }


                // --------------------------------------------
                // Minimum crossing interval.
                // --------------------------------------------

                if (
                    last_crossing < 0.0f
                    ||
                    (
                        crossing -
                        last_crossing
                    )
                    >=
                    MIN_CROSSING_INTERVAL
                )
                {
                    if (
                        crossing_count <
                        MAX_CROSSINGS
                    )
                    {
                        crossing_positions[
                            crossing_count
                        ] =
                            crossing;

                        crossing_count++;
                    }

                    last_crossing =
                        crossing;

                    // Re-arm only after the waveform goes
                    // negative again.
                    state =
                        0;
                }
            }
        }


        // Update filter state.
        previous_filtered =
            current_filtered;
    }

    return crossing_count;
}


// ------------------------------------------------------------
// Linear interpolation of the raw waveform
// ------------------------------------------------------------

static float interpolate_raw(
    const float *x,
    int count,
    float position
)
{
    if (
        position <=
        0.0f
    )
    {
        return x[0];
    }

    if (
        position >=
        (float)(count - 1)
    )
    {
        return x[
            count - 1
        ];
    }

    int i =
        (int)floorf(
            position
        );

    float frac =
        position -
        (float)i;

    return
        x[i]
        +
        frac *
        (
            x[i + 1] -
            x[i]
        );
}


// ------------------------------------------------------------
// Resample one complete RAW cycle to exactly 1000 samples.
//
// Important:
//
// The filtered waveform is used ONLY for timing.
//
// The actual cycle sent to the feature extractor is taken from
// the original raw waveform.
// ------------------------------------------------------------

static void resample_cycle(
    const float *raw,
    int raw_count,
    float start,
    float end,
    float *output
)
{
    float span =
        end -
        start;

    for (
        int i = 0;
        i < TARGET_CYCLE_LEN;
        i++
    )
    {
        float position =
            start
            +
            (
                (float)i /
                (float)TARGET_CYCLE_LEN
            )
            *
            span;

        output[i] =
            interpolate_raw(
                raw,
                raw_count,
                position
            );
    }
}


// ------------------------------------------------------------
// Run one frequency test
// ------------------------------------------------------------

static bool run_test(
    float frequency
)
{
    Serial.println();
    Serial.println(
        "------------------------------------------------"
    );

    Serial.printf(
        "TEST FREQUENCY: %.1f Hz\n",
        frequency
    );

    Serial.println(
        "------------------------------------------------"
    );


    // --------------------------------------------------------
    // Generate:
    //
    // 5% noise
    // 20% 3rd harmonic
    // 10% 5th harmonic
    // --------------------------------------------------------

    int sample_count =
        generate_waveform(
            frequency,
            0.05f,
            0.20f,
            0.10f
        );


    if (
        sample_count <= 0
    )
    {
        Serial.println(
            "ERROR: waveform generation failed."
        );

        return false;
    }


    Serial.printf(
        "Raw samples        : %d\n",
        sample_count
    );


    // --------------------------------------------------------
    // Detect robust crossings.
    // --------------------------------------------------------

    int crossing_count =
        detect_crossings(
            raw_samples,
            sample_count
        );


    Serial.printf(
        "Crossings           : %d\n",
        crossing_count
    );


    if (
        crossing_count <
        3
    )
    {
        Serial.println(
            "RESULT: FAIL"
        );

        Serial.println(
            "Reason: insufficient crossings."
        );

        return false;
    }


    // --------------------------------------------------------
    // Measure period.
    // --------------------------------------------------------

    float period_sum =
        0.0f;

    float period_min =
        1e9f;

    float period_max =
        0.0f;


    for (
        int i = 1;
        i < crossing_count;
        i++
    )
    {
        float period =
            crossing_positions[i]
            -
            crossing_positions[i - 1];

        period_sum +=
            period;

        if (
            period <
            period_min
        )
        {
            period_min =
                period;
        }

        if (
            period >
            period_max
        )
        {
            period_max =
                period;
        }
    }


    int detected_cycles =
        crossing_count - 1;


    float mean_period =
        period_sum /
        (float)detected_cycles;


    float measured_frequency =
        FS_HZ /
        mean_period;


    float frequency_error =
        100.0f *
        (
            measured_frequency -
            frequency
        )
        /
        frequency;


    Serial.printf(
        "Detected cycles     : %d\n",
        detected_cycles
    );

    Serial.printf(
        "Mean period         : %.4f samples\n",
        mean_period
    );

    Serial.printf(
        "Min period          : %.4f samples\n",
        period_min
    );

    Serial.printf(
        "Max period          : %.4f samples\n",
        period_max
    );

    Serial.printf(
        "Measured frequency  : %.6f Hz\n",
        measured_frequency
    );

    Serial.printf(
        "Frequency error     : %+.6f %%\n",
        frequency_error
    );


    // --------------------------------------------------------
    // Resample first complete cycle.
    //
    // Use RAW waveform, not filtered waveform.
    // --------------------------------------------------------

    uint32_t resample_start =
        micros();

    resample_cycle(
        raw_samples,
        sample_count,
        crossing_positions[0],
        crossing_positions[1],
        resampled_cycle
    );

    uint32_t resample_end =
        micros();


    // --------------------------------------------------------
    // Extract verified 8 features.
    // --------------------------------------------------------

    float features[8];

    uint32_t dsp_start =
        micros();

    extract_features_8(
        resampled_cycle,
        features
    );

    uint32_t dsp_end =
        micros();


    // --------------------------------------------------------
    // Run verified MLP.
    // --------------------------------------------------------

    float probability =
        0.0f;

    uint32_t mlp_start =
        micros();

    int prediction =
        predict_arc_8(
            features,
            &probability
        );

    uint32_t mlp_end =
        micros();


    // --------------------------------------------------------
    // Display resulting features.
    // --------------------------------------------------------

    Serial.println();
    Serial.println(
        "FIRST RESAMPLED CYCLE"
    );

    Serial.println(
        "------------------------------------------------"
    );

    Serial.printf(
        "Length              : %d\n",
        TARGET_CYCLE_LEN
    );

    Serial.printf(
        "crest               : %.8f\n",
        features[0]
    );

    Serial.printf(
        "skew                : %.8f\n",
        features[1]
    );

    Serial.printf(
        "kurt                : %.8f\n",
        features[2]
    );

    Serial.printf(
        "d1_ratio            : %.8f\n",
        features[3]
    );

    Serial.printf(
        "d2_ratio            : %.8f\n",
        features[4]
    );

    Serial.printf(
        "peak_asym           : %.8f\n",
        features[5]
    );

    Serial.printf(
        "energy_asym         : %.8f\n",
        features[6]
    );

    Serial.printf(
        "frac_small          : %.8f\n",
        features[7]
    );


    // --------------------------------------------------------
    // Display MLP result.
    // --------------------------------------------------------

    Serial.println();
    Serial.println(
        "MLP INFERENCE"
    );

    Serial.println(
        "------------------------------------------------"
    );

    Serial.printf(
        "Arc probability     : %.8f\n",
        probability
    );

    Serial.printf(
        "Prediction          : %s\n",
        prediction
            ? "ARC"
            : "NORMAL"
    );


    // --------------------------------------------------------
    // Timing.
    // --------------------------------------------------------

    Serial.println();
    Serial.println(
        "TIMING"
    );

    Serial.println(
        "------------------------------------------------"
    );

    Serial.printf(
        "Resampling time     : %lu us\n",
        (unsigned long)(
            resample_end -
            resample_start
        )
    );

    Serial.printf(
        "DSP time            : %lu us\n",
        (unsigned long)(
            dsp_end -
            dsp_start
        )
    );

    Serial.printf(
        "MLP time            : %lu us\n",
        (unsigned long)(
            mlp_end -
            mlp_start
        )
    );


    // --------------------------------------------------------
    // Test criterion.
    //
    // We only validate:
    //   - enough cycles detected
    //   - frequency error <= 0.5%
    //   - fixed cycle length
    //
    // ML prediction itself is not treated as a performance
    // metric for these artificial signals.
    // --------------------------------------------------------

    bool pass =
        (
            detected_cycles >=
            10
        )
        &&
        (
            fabsf(
                frequency_error
            )
            <=
            0.5f
        )
        &&
        (
            TARGET_CYCLE_LEN ==
            1000
        );


    Serial.println();

    if (pass)
    {
        Serial.println(
            "RESULT: PASS"
        );
    }
    else
    {
        Serial.println(
            "RESULT: FAIL"
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
        "================================================"
    );

    Serial.println(
        "ESP32 ROBUST CYCLE-SLICER ON-TARGET TEST"
    );

    Serial.println(
        "================================================"
    );


    Serial.printf(
        "CPU frequency : %d MHz\n",
        getCpuFrequencyMhz()
    );

    Serial.printf(
        "Arduino core  : %s\n",
        ESP_ARDUINO_VERSION_STR
    );

    Serial.printf(
        "ESP-IDF       : %s\n",
        esp_get_idf_version()
    );

    Serial.printf(
        "Sample rate   : %.0f Hz\n",
        FS_HZ
    );

    Serial.printf(
        "Low-pass      : %.1f Hz\n",
        LOWPASS_CUTOFF_HZ
    );

    Serial.printf(
        "Hysteresis    : %.2f\n",
        HYSTERESIS
    );

    Serial.printf(
        "Minimum gap   : %.0f samples\n",
        MIN_CROSSING_INTERVAL
    );


    bool all_pass =
        true;


    bool result;


    // 49.5 Hz
    result =
        run_test(
            49.5f
        );

    if (!result)
    {
        all_pass =
            false;
    }


    // 50.0 Hz
    result =
        run_test(
            50.0f
        );

    if (!result)
    {
        all_pass =
            false;
    }


    // 50.5 Hz
    result =
        run_test(
            50.5f
        );

    if (!result)
    {
        all_pass =
            false;
    }


    // --------------------------------------------------------
    // Final status.
    // --------------------------------------------------------

    Serial.println();
    Serial.println(
        "================================================"
    );

    if (all_pass)
    {
        Serial.println(
            "STATUS: ESP32 ROBUST SLICER TEST PASSED"
        );
    }
    else
    {
        Serial.println(
            "STATUS: ESP32 ROBUST SLICER TEST FAILED"
        );
    }

    Serial.println(
        "================================================"
    );
}


void loop()
{
    delay(
        5000
    );
}