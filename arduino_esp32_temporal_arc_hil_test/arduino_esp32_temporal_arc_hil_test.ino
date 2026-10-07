#include <Arduino.h>
#include <math.h>

#include "driver/adc.h"
#include "esp_adc/adc_continuous.h"
#include "esp_err.h"

extern "C" {
#include "feature_extraction.h"
}

#include "tinyml_engine_b.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ============================================================
// ESP32 TEMPORAL ARC-SIDE HIL TEST
//
// SAFE SYNTHETIC TEST ONLY
//
// Hardware:
//   GPIO25 DAC -> GPIO34 ADC
//
// Pattern:
//
//   Cycle 1 : CLEAN
//   Cycle 2 : CLEAN
//
//   Cycle 3 : DISTORTED / ARC-LIKE SYNTHETIC
//   Cycle 4 : DISTORTED / ARC-LIKE SYNTHETIC
//   Cycle 5 : DISTORTED / ARC-LIKE SYNTHETIC
//
// Voting:
//   3 ARC votes out of 5 -> final ARC
//
// IMPORTANT:
//   This does NOT represent a real electrical arc.
//   It only verifies that the embedded temporal voter can
//   accumulate three positive cycle decisions.
// ============================================================


// ------------------------------------------------------------
// DAC configuration
// ------------------------------------------------------------

#define DAC_PIN                 25

#define DAC_UPDATE_HZ           1000
#define DAC_WAVEFORM_HZ         50.0f

#define DAC_MIDPOINT            128.0f
#define DAC_AMPLITUDE           60.0f


// ------------------------------------------------------------
// DAC cycle structure
//
// At 1000 updates/sec and 50 Hz:
//
//   1000 / 50 = 20 DAC samples/cycle
// ------------------------------------------------------------

#define DAC_SAMPLES_PER_CYCLE   20


// First two cycles clean.
// Next three cycles distorted.
#define CLEAN_CYCLES            2
#define DISTORTED_CYCLES        3


// ------------------------------------------------------------
// Distortion configuration
//
// Synthetic only.
//
// Fundamental amplitude = 1.0
//
// Distorted cycles add:
//   20% 3rd harmonic
//   10% 5th harmonic
//   5% deterministic noise
//
// Plus a stronger high-frequency component to make the test
// waveform more visibly non-sinusoidal.
// ------------------------------------------------------------

#define H3_AMPLITUDE            0.20f
#define H5_AMPLITUDE            0.10f
#define HF_AMPLITUDE            0.15f

#define HF_FREQUENCY_HZ         400.0f

#define NOISE_LEVEL             0.05f


// ------------------------------------------------------------
// ADC configuration
// ------------------------------------------------------------

#define ADC_PIN                 34
#define ADC_CHANNEL             ADC_CHANNEL_6

#define SAMPLE_RATE_HZ          50000

// 7000 samples = 140 ms nominal.
// Enough for five complete 50 Hz cycles with margin.
#define ACQUISITION_SAMPLES     7000

#define FRAME_BYTES             256


// ------------------------------------------------------------
// Cycle slicing
// ------------------------------------------------------------

#define TARGET_CYCLE_LEN        1000

#define MAX_CROSSINGS           16

#define LOWPASS_CUTOFF_HZ       100.0f

#define HYSTERESIS              0.10f

#define MIN_CROSSING_INTERVAL   700.0f


// ------------------------------------------------------------
// Voting
// ------------------------------------------------------------

#define VOTE_CYCLES             5

#define ARC_VOTE_THRESHOLD      3

#define PROBABILITY_THRESHOLD   0.5f


// ------------------------------------------------------------
// Buffers
// ------------------------------------------------------------

static uint16_t adc_samples[
    ACQUISITION_SAMPLES
];

static float resampled_cycle[
    TARGET_CYCLE_LEN
];

static float crossing_positions[
    MAX_CROSSINGS
];

static float cycle_probabilities[
    VOTE_CYCLES
];

static int cycle_predictions[
    VOTE_CYCLES
];


// ------------------------------------------------------------
// ADC handle
// ------------------------------------------------------------

static adc_continuous_handle_t adc_handle =
    NULL;


// ------------------------------------------------------------
// DAC task
// ------------------------------------------------------------

static volatile bool waveform_running =
    false;

static TaskHandle_t dac_task_handle =
    NULL;


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
// ------------------------------------------------------------

static float deterministic_noise()
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

    return (
        u *
        NOISE_LEVEL /
        0.5773503f
    );
}


// ------------------------------------------------------------
// Determine whether current synthetic cycle is distorted
// ------------------------------------------------------------

static bool is_distorted_cycle(
    uint32_t cycle_number
)
{
    return (
        cycle_number >= CLEAN_CYCLES &&
        cycle_number <
            (
                CLEAN_CYCLES +
                DISTORTED_CYCLES
            )
    );
}


// ------------------------------------------------------------
// Generate one DAC sample
// ------------------------------------------------------------

static uint8_t generate_dac_sample(
    uint32_t sample_index
)
{
    uint32_t cycle_number =
        sample_index /
        DAC_SAMPLES_PER_CYCLE;


    uint32_t sample_in_cycle =
        sample_index %
        DAC_SAMPLES_PER_CYCLE;


    float phase =
        2.0f *
        (float)M_PI *
        (
            (float)sample_in_cycle /
            (float)DAC_SAMPLES_PER_CYCLE
        );


    float value =
        sinf(phase);


    bool distorted =
        is_distorted_cycle(
            cycle_number
        );


    if (distorted)
    {
        // 3rd harmonic.
        value +=
            H3_AMPLITUDE *
            sinf(
                3.0f *
                phase
            );


        // 5th harmonic.
        value +=
            H5_AMPLITUDE *
            sinf(
                5.0f *
                phase
            );


        // Stronger high-frequency component.
        //
        // phase corresponds to 50 Hz, so:
        //
        // 400 / 50 = 8
        //
        value +=
            HF_AMPLITUDE *
            sinf(
                8.0f *
                phase
            );


        // Deterministic broadband component.
        value +=
            deterministic_noise();
    }


    float dac_value =
        DAC_MIDPOINT +
        DAC_AMPLITUDE *
        value;


    if (dac_value < 0.0f)
    {
        dac_value = 0.0f;
    }


    if (dac_value > 255.0f)
    {
        dac_value = 255.0f;
    }


    return (
        (uint8_t)dac_value
    );
}


// ------------------------------------------------------------
// DAC waveform task
// ------------------------------------------------------------

static void dac_waveform_task(
    void *parameter
)
{
    (void)parameter;


    uint32_t sample_index =
        0;


    noise_state =
        0x12345678UL;


    while (
        waveform_running
    )
    {
        uint8_t value =
            generate_dac_sample(
                sample_index
            );


        dacWrite(
            DAC_PIN,
            value
        );


        sample_index++;


        vTaskDelay(
            pdMS_TO_TICKS(1)
        );
    }


    dacWrite(
        DAC_PIN,
        (uint8_t)DAC_MIDPOINT
    );


    dac_task_handle =
        NULL;


    vTaskDelete(
        NULL
    );
}


// ------------------------------------------------------------
// Start DAC task
// ------------------------------------------------------------

static bool start_waveform()
{
    waveform_running =
        true;


    BaseType_t result =
        xTaskCreatePinnedToCore(
            dac_waveform_task,
            "DACWaveform",
            2048,
            NULL,
            1,
            &dac_task_handle,
            0
        );


    if (
        result !=
        pdPASS
    )
    {
        waveform_running =
            false;

        dac_task_handle =
            NULL;

        return false;
    }


    return true;
}


// ------------------------------------------------------------
// Stop DAC task
// ------------------------------------------------------------

static void stop_waveform()
{
    waveform_running =
        false;


    for (
        int i = 0;
        i < 20 &&
        dac_task_handle != NULL;
        i++
    )
    {
        delay(1);
    }


    dacWrite(
        DAC_PIN,
        (uint8_t)DAC_MIDPOINT
    );
}


// ------------------------------------------------------------
// Start ADC DMA
// ------------------------------------------------------------

static bool start_adc()
{
    adc_continuous_handle_cfg_t handle_config = {};

    handle_config.max_store_buf_size =
        8192;

    handle_config.conv_frame_size =
        FRAME_BYTES;


    esp_err_t err =
        adc_continuous_new_handle(
            &handle_config,
            &adc_handle
        );


    if (
        err !=
        ESP_OK
    )
    {
        Serial.printf(
            "ERROR ADC handle: 0x%X\n",
            err
        );

        adc_handle =
            NULL;

        return false;
    }


    adc_digi_pattern_config_t pattern = {};

    pattern.atten =
        ADC_ATTEN_DB_11;

    pattern.channel =
        ADC_CHANNEL;

    pattern.unit =
        ADC_UNIT_1;

    pattern.bit_width =
        ADC_BITWIDTH_12;


    adc_continuous_config_t config = {};

    config.sample_freq_hz =
        SAMPLE_RATE_HZ;

    config.conv_mode =
        ADC_CONV_SINGLE_UNIT_1;

    config.format =
        ADC_DIGI_OUTPUT_FORMAT_TYPE1;

    config.pattern_num =
        1;

    config.adc_pattern =
        &pattern;


    err =
        adc_continuous_config(
            adc_handle,
            &config
        );


    if (
        err !=
        ESP_OK
    )
    {
        Serial.printf(
            "ERROR ADC config: 0x%X\n",
            err
        );

        adc_continuous_deinit(
            adc_handle
        );

        adc_handle =
            NULL;

        return false;
    }


    err =
        adc_continuous_start(
            adc_handle
        );


    if (
        err !=
        ESP_OK
    )
    {
        Serial.printf(
            "ERROR ADC start: 0x%X\n",
            err
        );

        adc_continuous_deinit(
            adc_handle
        );

        adc_handle =
            NULL;

        return false;
    }


    return true;
}


// ------------------------------------------------------------
// Flush startup DMA
// ------------------------------------------------------------

static void flush_startup_data()
{
    uint8_t dma_buffer[
        FRAME_BYTES
    ];


    while (true)
    {
        uint32_t bytes_read =
            0;


        esp_err_t err =
            adc_continuous_read(
                adc_handle,
                dma_buffer,
                sizeof(dma_buffer),
                &bytes_read,
                0
            );


        if (
            err ==
            ESP_ERR_TIMEOUT
        )
        {
            break;
        }


        if (
            err !=
            ESP_OK
        )
        {
            break;
        }
    }
}


// ------------------------------------------------------------
// Acquire 7000 samples
// ------------------------------------------------------------

static bool acquire_samples(
    uint32_t &elapsed_us
)
{
    uint8_t dma_buffer[
        FRAME_BYTES
    ];


    uint32_t sample_count =
        0;


    uint32_t start_us =
        micros();


    while (
        sample_count <
        ACQUISITION_SAMPLES
    )
    {
        uint32_t bytes_read =
            0;


        esp_err_t err =
            adc_continuous_read(
                adc_handle,
                dma_buffer,
                sizeof(dma_buffer),
                &bytes_read,
                100
            );


        if (
            err ==
            ESP_ERR_TIMEOUT
        )
        {
            continue;
        }


        if (
            err !=
            ESP_OK
        )
        {
            Serial.printf(
                "ERROR ADC read: 0x%X\n",
                err
            );

            return false;
        }


        for (
            uint32_t i = 0;
            i + 1 < bytes_read &&
            sample_count <
                ACQUISITION_SAMPLES;
            i += 2
        )
        {
            uint16_t word =
                (
                    (uint16_t)
                    dma_buffer[i]
                )
                |
                (
                    (
                        (uint16_t)
                        dma_buffer[i + 1]
                    )
                    << 8
                );


            uint16_t channel =
                (
                    word >> 12
                )
                &
                0x0F;


            uint16_t raw =
                word &
                0x0FFF;


            if (
                channel ==
                ADC_CHANNEL
            )
            {
                adc_samples[
                    sample_count
                ] =
                    raw;

                sample_count++;
            }
        }
    }


    elapsed_us =
        micros() -
        start_us;


    return (
        sample_count ==
        ACQUISITION_SAMPLES
    );
}


// ------------------------------------------------------------
// Calculate mean
// ------------------------------------------------------------

static float calculate_mean()
{
    uint64_t sum =
        0;


    for (
        int i = 0;
        i < ACQUISITION_SAMPLES;
        i++
    )
    {
        sum +=
            adc_samples[i];
    }


    return (
        (float)sum /
        (float)ACQUISITION_SAMPLES
    );
}


// ------------------------------------------------------------
// Calculate peak centered amplitude
// ------------------------------------------------------------

static float calculate_signal_scale(
    float dc_mean
)
{
    float max_abs =
        0.0f;


    for (
        int i = 0;
        i < ACQUISITION_SAMPLES;
        i++
    )
    {
        float centered =
            (
                (float)
                adc_samples[i]
                -
                dc_mean
            );


        float magnitude =
            fabsf(
                centered
            );


        if (
            magnitude >
            max_abs
        )
        {
            max_abs =
                magnitude;
        }
    }


    if (
        max_abs <
        1e-6f
    )
    {
        max_abs =
            1.0f;
    }


    return max_abs;
}


// ------------------------------------------------------------
// Robust zero crossing
// ------------------------------------------------------------

static int detect_crossings(
    float dc_mean,
    float signal_scale
)
{
    int crossing_count =
        0;


    int state =
        0;


    float last_crossing =
        -1.0f;


    float dt =
        1.0f /
        (float)SAMPLE_RATE_HZ;


    float rc =
        1.0f /
        (
            2.0f *
            (float)M_PI *
            LOWPASS_CUTOFF_HZ
        );


    float alpha =
        dt /
        (rc + dt);


    float first_centered =
        (
            (
                (float)
                adc_samples[0]
                -
                dc_mean
            )
            /
            signal_scale
        );


    float previous_filtered =
        first_centered;


    for (
        int i = 1;
        i < ACQUISITION_SAMPLES;
        i++
    )
    {
        float centered =
            (
                (
                    (float)
                    adc_samples[i]
                    -
                    dc_mean
                )
                /
                signal_scale
            );


        float current_filtered =
            previous_filtered
            +
            alpha *
            (
                centered
                -
                previous_filtered
            );


        float previous =
            previous_filtered;


        float current =
            current_filtered;


        if (
            state == 0
        )
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


                if (
                    fabsf(
                        denominator
                    )
                    <
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


                if (
                    last_crossing <
                    0.0f
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


                    state =
                        0;
                }
            }
        }


        previous_filtered =
            current_filtered;
    }


    return crossing_count;
}


// ------------------------------------------------------------
// Raw interpolation
// ------------------------------------------------------------

static float interpolate_raw(
    float position,
    float dc_mean
)
{
    if (
        position <=
        0.0f
    )
    {
        return (
            (float)
            adc_samples[0]
            -
            dc_mean
        );
    }


    if (
        position >=
        (float)(
            ACQUISITION_SAMPLES -
            1
        )
    )
    {
        return (
            (float)
            adc_samples[
                ACQUISITION_SAMPLES -
                1
            ]
            -
            dc_mean
        );
    }


    int index =
        (int)floorf(
            position
        );


    float fraction =
        position -
        (float)index;


    float x0 =
        (
            (float)
            adc_samples[index]
            -
            dc_mean
        );


    float x1 =
        (
            (float)
            adc_samples[index + 1]
            -
            dc_mean
        );


    return (
        x0
        +
        fraction *
        (
            x1 -
            x0
        )
    );
}


// ------------------------------------------------------------
// Resample cycle
// ------------------------------------------------------------

static void resample_cycle(
    float start,
    float end,
    float dc_mean
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


        resampled_cycle[i] =
            interpolate_raw(
                position,
                dc_mean
            );
    }
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
        "=================================================="
    );

    Serial.println(
        "ESP32 SYNTHETIC ARC-SIDE 3-OF-5 HIL TEST"
    );

    Serial.println(
        "=================================================="
    );


    Serial.println(
        "D25 -> D34 required"
    );


    Serial.println(
        "NO MAINS"
    );


    Serial.println();


    // --------------------------------------------------------
    // Start DAC
    // --------------------------------------------------------

    if (
        !start_waveform()
    )
    {
        Serial.println(
            "STATUS: DAC START FAILED"
        );

        while (true)
        {
            delay(1000);
        }
    }


    // Let the waveform settle.
    delay(
        200
    );


    // --------------------------------------------------------
    // Start ADC
    // --------------------------------------------------------

    if (
        !start_adc()
    )
    {
        stop_waveform();


        Serial.println(
            "STATUS: ADC START FAILED"
        );


        while (true)
        {
            delay(1000);
        }
    }


    flush_startup_data();


    // --------------------------------------------------------
    // Acquire.
    // --------------------------------------------------------

    uint32_t acquisition_us =
        0;


    bool acquired =
        acquire_samples(
            acquisition_us
        );


    adc_continuous_stop(
        adc_handle
    );


    adc_continuous_deinit(
        adc_handle
    );


    adc_handle =
        NULL;


    stop_waveform();


    if (!acquired)
    {
        Serial.println(
            "STATUS: ADC ACQUISITION FAILED"
        );


        while (true)
        {
            delay(1000);
        }
    }


    float acquisition_ms =
        acquisition_us /
        1000.0f;


    float observed_rate =
        (
            (float)
            ACQUISITION_SAMPLES
            /
            (float)
            acquisition_us
        )
        *
        1000000.0f;


    // --------------------------------------------------------
    // Statistics.
    // --------------------------------------------------------

    float dc_mean =
        calculate_mean();


    float signal_scale =
        calculate_signal_scale(
            dc_mean
        );


    Serial.println();

    Serial.println(
        "ACQUISITION"
    );

    Serial.println(
        "----------------------------------------"
    );


    Serial.printf(
        "Elapsed            : %.3f ms\n",
        acquisition_ms
    );


    Serial.printf(
        "Observed rate      : %.2f samples/sec\n",
        observed_rate
    );


    Serial.printf(
        "Mean ADC           : %.3f\n",
        dc_mean
    );


    Serial.printf(
        "Peak centered ADC  : %.3f\n",
        signal_scale
    );


    // --------------------------------------------------------
    // Cycle detection.
    // --------------------------------------------------------

    uint32_t slicer_start =
        micros();


    int crossing_count =
        detect_crossings(
            dc_mean,
            signal_scale
        );


    uint32_t slicer_end =
        micros();


    Serial.println();

    Serial.println(
        "CYCLE DETECTION"
    );

    Serial.println(
        "----------------------------------------"
    );


    Serial.printf(
        "Crossings detected : %d\n",
        crossing_count
    );


    Serial.printf(
        "Slicer time        : %lu us\n",
        (unsigned long)(
            slicer_end -
            slicer_start
        )
    );


    if (
        crossing_count <
        (
            VOTE_CYCLES + 1
        )
    )
    {
        Serial.println();

        Serial.println(
            "STATUS: NOT ENOUGH COMPLETE CYCLES"
        );

        Serial.println(
            "Expected at least 6 rising crossings."
        );


        while (true)
        {
            delay(5000);
        }
    }


    // --------------------------------------------------------
    // Process exactly five cycles.
    // --------------------------------------------------------

    int arc_votes =
        0;


    uint32_t total_processing_us =
        0;


    Serial.println();

    Serial.println(
        "PER-CYCLE RESULTS"
    );

    Serial.println(
        "----------------------------------------"
    );


    for (
        int cycle = 0;
        cycle < VOTE_CYCLES;
        cycle++
    )
    {
        float period =
            crossing_positions[
                cycle + 1
            ]
            -
            crossing_positions[
                cycle
            ];


        float frequency =
            (float)
            SAMPLE_RATE_HZ /
            period;


        // -----------------------------------------------
        // Resample.
        // -----------------------------------------------

        uint32_t cycle_start =
            micros();


        resample_cycle(
            crossing_positions[cycle],
            crossing_positions[cycle + 1],
            dc_mean
        );


        // -----------------------------------------------
        // Features.
        // -----------------------------------------------

        float features[8];


        extract_features_8(
            resampled_cycle,
            features
        );


        // -----------------------------------------------
        // MLP.
        // -----------------------------------------------

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


        cycle_probabilities[
            cycle
        ] =
            probability;


        cycle_predictions[
            cycle
        ] =
            prediction;


        total_processing_us +=
            processing_us;


        if (
            probability >=
            PROBABILITY_THRESHOLD
        )
        {
            arc_votes++;
        }


        bool expected_synthetic_arc =
            (
                cycle >=
                CLEAN_CYCLES
            );


        Serial.printf(
            "Cycle %d | "
            "Freq %.6f Hz | "
            "Prob %.8f | "
            "Pred %s | "
            "Synthetic %s | "
            "Time %lu us\n",
            cycle + 1,
            frequency,
            probability,
            prediction
                ? "ARC"
                : "NORMAL",
            expected_synthetic_arc
                ? "DISTORTED"
                : "CLEAN",
            (unsigned long)
                processing_us
        );
    }


    // --------------------------------------------------------
    // Final 3-of-5 decision.
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


    Serial.println();

    Serial.println(
        "TEMPORAL VOTING"
    );

    Serial.println(
        "----------------------------------------"
    );


    for (
        int i = 0;
        i < VOTE_CYCLES;
        i++
    )
    {
        Serial.printf(
            "Cycle %d probability : %.8f\n",
            i + 1,
            cycle_probabilities[i]
        );
    }


    Serial.printf(
        "ARC votes            : %d / %d\n",
        arc_votes,
        VOTE_CYCLES
    );


    Serial.printf(
        "Required ARC votes   : %d\n",
        ARC_VOTE_THRESHOLD
    );


    Serial.printf(
        "Mean probability     : %.8f\n",
        mean_probability
    );


    Serial.printf(
        "FINAL DECISION       : %s\n",
        final_arc
            ? "ARC"
            : "NORMAL"
    );


    Serial.println();

    Serial.println(
        "TIMING"
    );

    Serial.println(
        "----------------------------------------"
    );


    Serial.printf(
        "Acquisition          : %.3f ms\n",
        acquisition_ms
    );


    Serial.printf(
        "Slicer               : %lu us\n",
        (unsigned long)(
            slicer_end -
            slicer_start
        )
    );


    Serial.printf(
        "5-cycle processing   : %lu us\n",
        (unsigned long)
            total_processing_us
    );


    Serial.printf(
        "Average cycle proc   : %.1f us\n",
        (
            (float)
            total_processing_us
            /
            (float)VOTE_CYCLES
        )
    );


    Serial.println();

    Serial.println(
        "=================================================="
    );


    if (final_arc)
    {
        Serial.println(
            "STATUS: SYNTHETIC ARC-SIDE 3-OF-5 TEST PASSED"
        );
    }
    else
    {
        Serial.println(
            "STATUS: SYNTHETIC ARC-SIDE TEST DID NOT REACH 3-OF-5"
        );
    }


    Serial.println(
        "=================================================="
    );
}


void loop()
{
    delay(
        5000
    );
}