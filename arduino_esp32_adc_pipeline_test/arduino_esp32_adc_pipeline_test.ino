#include <Arduino.h>
#include <math.h>

#include "driver/adc.h"
#include "esp_adc/adc_continuous.h"
#include "esp_err.h"
#include "soc/soc_caps.h"

extern "C" {
#include "feature_extraction.h"
}

#include "tinyml_engine_b.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ============================================================
// ESP32 ADC -> ROBUST CYCLE SLICING -> 8 FEATURES -> MLP
//
// Board:
//   DOIT ESP32 DEVKIT V1 / classic ESP32
//
// ADC:
//   GPIO34 = ADC1_CH6
//
// Sampling:
//   50 kHz
//
// Acquisition:
//   5000 samples = nominally 100 ms
//
// Processing:
//   ADC DMA
//      |
//      +--> DC removal
//      |
//      +--> normalize timing signal
//      |
//      +--> low-pass filter
//      |
//      +--> hysteresis crossing detection
//      |
//      +--> minimum crossing interval
//      |
//      +--> slice RAW centered waveform
//      |
//      +--> resample to 1000 samples
//      |
//      +--> 8 verified features
//      |
//      +--> verified MLP
//
// IMPORTANT:
//   No mains.
//   For the current test GPIO34 may remain UNCONNECTED.
// ============================================================


// ------------------------------------------------------------
// ADC configuration
// ------------------------------------------------------------

#define ADC_PIN                 34
#define ADC_CHANNEL             ADC_CHANNEL_6

#define SAMPLE_RATE_HZ          50000
#define ACQUISITION_SAMPLES     5000
#define FRAME_BYTES             256


// ------------------------------------------------------------
// Cycle slicing configuration
// ------------------------------------------------------------

#define TARGET_CYCLE_LEN        1000
#define MAX_CROSSINGS           16

#define LOWPASS_CUTOFF_HZ       100.0f
#define HYSTERESIS              0.10f

// 14 ms at 50 kHz.
// A 50 Hz cycle is approximately 20 ms.
#define MIN_CROSSING_INTERVAL   700.0f


// ------------------------------------------------------------
// Buffers
// ------------------------------------------------------------

static uint16_t adc_samples[ACQUISITION_SAMPLES];

static float resampled_cycle[TARGET_CYCLE_LEN];

static float crossing_positions[MAX_CROSSINGS];


// ------------------------------------------------------------
// ADC handle
// ------------------------------------------------------------

static adc_continuous_handle_t adc_handle = NULL;


// ------------------------------------------------------------
// Start ADC DMA
// ------------------------------------------------------------

static bool start_adc()
{
    adc_continuous_handle_cfg_t handle_config = {};

    handle_config.max_store_buf_size = 8192;
    handle_config.conv_frame_size = FRAME_BYTES;

    esp_err_t err =
        adc_continuous_new_handle(
            &handle_config,
            &adc_handle
        );

    if (err != ESP_OK)
    {
        Serial.printf(
            "ERROR: new ADC handle: 0x%X\n",
            err
        );

        adc_handle = NULL;
        return false;
    }


    adc_digi_pattern_config_t pattern = {};

    pattern.atten = ADC_ATTEN_DB_11;
    pattern.channel = ADC_CHANNEL;
    pattern.unit = ADC_UNIT_1;
    pattern.bit_width = ADC_BITWIDTH_12;


    adc_continuous_config_t config = {};

    config.sample_freq_hz = SAMPLE_RATE_HZ;
    config.conv_mode = ADC_CONV_SINGLE_UNIT_1;
    config.format = ADC_DIGI_OUTPUT_FORMAT_TYPE1;
    config.pattern_num = 1;
    config.adc_pattern = &pattern;


    err =
        adc_continuous_config(
            adc_handle,
            &config
        );

    if (err != ESP_OK)
    {
        Serial.printf(
            "ERROR: ADC configuration: 0x%X\n",
            err
        );

        adc_continuous_deinit(adc_handle);
        adc_handle = NULL;

        return false;
    }


    err =
        adc_continuous_start(
            adc_handle
        );

    if (err != ESP_OK)
    {
        Serial.printf(
            "ERROR: ADC start: 0x%X\n",
            err
        );

        adc_continuous_deinit(adc_handle);
        adc_handle = NULL;

        return false;
    }


    return true;
}


// ------------------------------------------------------------
// Flush startup DMA data
// ------------------------------------------------------------

static void flush_startup_data()
{
    uint8_t dma_buffer[FRAME_BYTES];

    while (true)
    {
        uint32_t bytes_read = 0;

        esp_err_t err =
            adc_continuous_read(
                adc_handle,
                dma_buffer,
                sizeof(dma_buffer),
                &bytes_read,
                0
            );

        if (err == ESP_ERR_TIMEOUT)
        {
            break;
        }

        if (err != ESP_OK)
        {
            Serial.printf(
                "WARNING: startup flush read: 0x%X\n",
                err
            );

            break;
        }

        // Deliberately discard startup data.
    }
}


// ------------------------------------------------------------
// Discard one fresh DMA frame after startup flush
// ------------------------------------------------------------

static bool discard_one_frame()
{
    uint8_t dma_buffer[FRAME_BYTES];

    while (true)
    {
        uint32_t bytes_read = 0;

        esp_err_t err =
            adc_continuous_read(
                adc_handle,
                dma_buffer,
                sizeof(dma_buffer),
                &bytes_read,
                100
            );

        if (err == ESP_ERR_TIMEOUT)
        {
            continue;
        }

        if (err != ESP_OK)
        {
            Serial.printf(
                "ERROR: fresh-frame read: 0x%X\n",
                err
            );

            return false;
        }

        return true;
    }
}


// ------------------------------------------------------------
// Acquire exactly 5000 valid ADC samples
// ------------------------------------------------------------

static bool acquire_samples(
    uint32_t &elapsed_us
)
{
    uint8_t dma_buffer[FRAME_BYTES];

    uint32_t sample_count = 0;

    uint32_t start_us = micros();


    while (
        sample_count <
        ACQUISITION_SAMPLES
    )
    {
        uint32_t bytes_read = 0;

        esp_err_t err =
            adc_continuous_read(
                adc_handle,
                dma_buffer,
                sizeof(dma_buffer),
                &bytes_read,
                100
            );

        if (err == ESP_ERR_TIMEOUT)
        {
            continue;
        }

        if (err != ESP_OK)
        {
            Serial.printf(
                "ERROR: ADC read: 0x%X\n",
                err
            );

            return false;
        }


        for (
            uint32_t i = 0;
            i + 1 < bytes_read &&
            sample_count < ACQUISITION_SAMPLES;
            i += 2
        )
        {
            uint16_t word =
                (uint16_t)dma_buffer[i]
                |
                (
                    (uint16_t)dma_buffer[i + 1]
                    << 8
                );


            uint16_t channel =
                (word >> 12) & 0x0F;


            uint16_t raw =
                word & 0x0FFF;


            if (channel == ADC_CHANNEL)
            {
                adc_samples[sample_count] =
                    raw;

                sample_count++;
            }
        }
    }


    elapsed_us =
        micros() - start_us;


    return (
        sample_count ==
        ACQUISITION_SAMPLES
    );
}


// ------------------------------------------------------------
// Calculate DC mean
// ------------------------------------------------------------

static float calculate_mean()
{
    uint64_t sum = 0;

    for (
        int i = 0;
        i < ACQUISITION_SAMPLES;
        i++
    )
    {
        sum += adc_samples[i];
    }

    return (
        (float)sum /
        (float)ACQUISITION_SAMPLES
    );
}


// ------------------------------------------------------------
// Calculate maximum absolute centered amplitude
// ------------------------------------------------------------

static float calculate_signal_scale(
    float dc_mean
)
{
    float max_abs = 0.0f;

    for (
        int i = 0;
        i < ACQUISITION_SAMPLES;
        i++
    )
    {
        float centered =
            (float)adc_samples[i] -
            dc_mean;

        float magnitude =
            fabsf(centered);

        if (magnitude > max_abs)
        {
            max_abs = magnitude;
        }
    }


    if (max_abs < 1e-6f)
    {
        max_abs = 1.0f;
    }

    return max_abs;
}


// ------------------------------------------------------------
// Robust hysteresis zero-crossing detector
//
// The filtered normalized signal is used ONLY for timing.
//
// The original centered ADC waveform is preserved for the
// actual feature extraction.
// ------------------------------------------------------------

static int detect_crossings(
    float dc_mean,
    float signal_scale
)
{
    int crossing_count = 0;

    int state = 0;

    float last_crossing = -1.0f;


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
            (float)adc_samples[0] -
            dc_mean
        )
        /
        signal_scale;


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
                (float)adc_samples[i] -
                dc_mean
            )
            /
            signal_scale;


        float current_filtered =
            previous_filtered +
            alpha *
            (
                centered -
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
                state = 1;
            }
        }


        // ----------------------------------------------------
        // State 1:
        // Wait for rising transition through +H.
        // ----------------------------------------------------

        else
        {
            if (
                previous < HYSTERESIS &&
                current >= HYSTERESIS
            )
            {
                float denominator =
                    current -
                    previous;


                float crossing;


                if (
                    fabsf(
                        denominator
                    ) < 1e-15f
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


                // ------------------------------------------------
                // Reject unrealistically close crossings.
                // ------------------------------------------------

                if (
                    last_crossing < 0.0f ||
                    (
                        crossing -
                        last_crossing
                    ) >=
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
                        ] = crossing;

                        crossing_count++;
                    }


                    last_crossing =
                        crossing;


                    // Re-arm only after the filtered signal
                    // returns through the negative region.
                    state = 0;
                }
            }
        }


        previous_filtered =
            current_filtered;
    }


    return crossing_count;
}


// ------------------------------------------------------------
// Interpolate RAW centered ADC waveform
// ------------------------------------------------------------

static float interpolate_raw(
    float position,
    float dc_mean
)
{
    if (position <= 0.0f)
    {
        return (
            (float)adc_samples[0] -
            dc_mean
        );
    }


    if (
        position >=
        (float)(
            ACQUISITION_SAMPLES - 1
        )
    )
    {
        return (
            (float)adc_samples[
                ACQUISITION_SAMPLES - 1
            ]
            -
            dc_mean
        );
    }


    int index =
        (int)floorf(position);


    float fraction =
        position -
        (float)index;


    float x0 =
        (
            (float)adc_samples[index] -
            dc_mean
        );


    float x1 =
        (
            (float)adc_samples[index + 1] -
            dc_mean
        );


    return (
        x0 +
        fraction *
        (
            x1 - x0
        )
    );
}


// ------------------------------------------------------------
// Resample one RAW cycle to exactly 1000 samples
// ------------------------------------------------------------

static void resample_cycle(
    float start,
    float end,
    float dc_mean
)
{
    float span =
        end - start;


    for (
        int i = 0;
        i < TARGET_CYCLE_LEN;
        i++
    )
    {
        float position =
            start +
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
// Print ADC statistics
// ------------------------------------------------------------

static void print_adc_statistics(
    float dc_mean,
    float signal_scale
)
{
    uint16_t min_value = 4095;
    uint16_t max_value = 0;


    for (
        int i = 0;
        i < ACQUISITION_SAMPLES;
        i++
    )
    {
        if (
            adc_samples[i] <
            min_value
        )
        {
            min_value =
                adc_samples[i];
        }


        if (
            adc_samples[i] >
            max_value
        )
        {
            max_value =
                adc_samples[i];
        }
    }


    Serial.println();
    Serial.println("ADC STATISTICS");
    Serial.println("----------------------------------------");


    Serial.printf(
        "Samples            : %d\n",
        ACQUISITION_SAMPLES
    );


    Serial.printf(
        "Nominal window     : %.2f ms\n",
        (
            1000.0f *
            (float)ACQUISITION_SAMPLES /
            (float)SAMPLE_RATE_HZ
        )
    );


    Serial.printf(
        "Min ADC            : %u\n",
        min_value
    );


    Serial.printf(
        "Max ADC            : %u\n",
        max_value
    );


    Serial.printf(
        "Mean ADC           : %.3f\n",
        dc_mean
    );


    Serial.printf(
        "Peak centered ADC  : %.3f\n",
        signal_scale
    );
}


// ------------------------------------------------------------
// SETUP
// ------------------------------------------------------------

void setup()
{
    Serial.begin(115200);

    delay(1000);


    Serial.println();
    Serial.println(
        "=================================================="
    );

    Serial.println(
        "ESP32 ADC + ROBUST CYCLE + MLP PIPELINE TEST"
    );

    Serial.println(
        "=================================================="
    );


    Serial.printf(
        "Board              : DOIT ESP32 DEVKIT V1\n"
    );


    Serial.printf(
        "CPU frequency      : %d MHz\n",
        getCpuFrequencyMhz()
    );


    Serial.printf(
        "Arduino core       : %s\n",
        ESP_ARDUINO_VERSION_STR
    );


    Serial.printf(
        "ESP-IDF            : %s\n",
        esp_get_idf_version()
    );


    Serial.printf(
        "ADC pin            : GPIO%d\n",
        ADC_PIN
    );


    Serial.println(
        "ADC channel        : ADC1_CH6"
    );


    Serial.printf(
        "Configured rate    : %d Hz\n",
        SAMPLE_RATE_HZ
    );


    Serial.printf(
        "Samples             : %d\n",
        ACQUISITION_SAMPLES
    );


    Serial.printf(
        "Nominal window     : %.2f ms\n",
        (
            1000.0f *
            (float)ACQUISITION_SAMPLES /
            (float)SAMPLE_RATE_HZ
        )
    );


    Serial.println();


    Serial.println(
        "IMPORTANT:"
    );


    Serial.println(
        "GPIO34 may remain UNCONNECTED for this run."
    );


    Serial.println(
        "Do NOT connect mains."
    );


    // --------------------------------------------------------
    // Start ADC
    // --------------------------------------------------------

    if (!start_adc())
    {
        Serial.println();
        Serial.println(
            "STATUS: ADC START FAILED"
        );

        while (true)
        {
            delay(1000);
        }
    }


    Serial.println();
    Serial.println(
        "ADC DMA started."
    );


    // --------------------------------------------------------
    // Flush startup DMA data
    // --------------------------------------------------------

    Serial.println(
        "Flushing startup DMA data..."
    );

    flush_startup_data();

    Serial.println(
        "Startup DMA data flushed."
    );


    // --------------------------------------------------------
    // Discard one fresh DMA frame
    // --------------------------------------------------------

    Serial.println(
        "Discarding one fresh DMA frame..."
    );


    if (!discard_one_frame())
    {
        Serial.println();
        Serial.println(
            "STATUS: DMA FRAME PREPARATION FAILED"
        );


        adc_continuous_stop(
            adc_handle
        );

        adc_continuous_deinit(
            adc_handle
        );

        adc_handle = NULL;


        while (true)
        {
            delay(1000);
        }
    }


    Serial.println(
        "Fresh DMA frame discarded."
    );


    // --------------------------------------------------------
    // Acquire 5000 samples
    // --------------------------------------------------------

    Serial.println();
    Serial.println(
        "Starting clean 5000-sample acquisition..."
    );


    uint32_t acquisition_us = 0;


    bool acquired =
        acquire_samples(
            acquisition_us
        );


    // Stop ADC.
    adc_continuous_stop(
        adc_handle
    );

    adc_continuous_deinit(
        adc_handle
    );

    adc_handle = NULL;


    if (!acquired)
    {
        Serial.println();
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


    float observed_throughput =
        (
            (float)ACQUISITION_SAMPLES /
            (float)acquisition_us
        )
        *
        1000000.0f;


    // --------------------------------------------------------
    // Statistics
    // --------------------------------------------------------

    float dc_mean =
        calculate_mean();


    float signal_scale =
        calculate_signal_scale(
            dc_mean
        );


    print_adc_statistics(
        dc_mean,
        signal_scale
    );


    // --------------------------------------------------------
    // Acquisition timing
    // --------------------------------------------------------

    Serial.println();

    Serial.println(
        "ACQUISITION TIMING"
    );

    Serial.println(
        "----------------------------------------"
    );


    Serial.printf(
        "Elapsed             : %.3f ms\n",
        acquisition_ms
    );


    Serial.printf(
        "Observed throughput : %.2f samples/sec\n",
        observed_throughput
    );


    Serial.printf(
        "Configured rate     : %d samples/sec\n",
        SAMPLE_RATE_HZ
    );


    Serial.printf(
        "Short-window error  : %+.3f %%\n",
        (
            100.0f *
            (
                observed_throughput -
                (float)SAMPLE_RATE_HZ
            )
            /
            (float)SAMPLE_RATE_HZ
        )
    );


    Serial.println(
        "NOTE: authoritative ADC-rate result remains the"
    );

    Serial.println(
        "previously verified 60-second DMA measurement."
    );


    // --------------------------------------------------------
    // Robust cycle detection
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


    // --------------------------------------------------------
    // Need at least one complete cycle.
    // --------------------------------------------------------

    if (crossing_count < 2)
    {
        Serial.println();

        Serial.println(
            "STATUS: NO COMPLETE CYCLE DETECTED"
        );


        Serial.println(
            "ADC acquisition completed successfully."
        );


        Serial.println(
            "A safe periodic analog signal is required"
        );


        Serial.println(
            "before cycle slicing and MLP inference."
        );


        while (true)
        {
            delay(5000);
        }
    }


    // --------------------------------------------------------
    // First cycle period
    // --------------------------------------------------------

    float first_period =
        crossing_positions[1] -
        crossing_positions[0];


    float first_frequency =
        (float)SAMPLE_RATE_HZ /
        first_period;


    Serial.printf(
        "First period       : %.3f samples\n",
        first_period
    );


    Serial.printf(
        "First frequency    : %.6f Hz\n",
        first_frequency
    );


    // --------------------------------------------------------
    // Resample first raw cycle
    // --------------------------------------------------------

    uint32_t resample_start =
        micros();


    resample_cycle(
        crossing_positions[0],
        crossing_positions[1],
        dc_mean
    );


    uint32_t resample_end =
        micros();


    // --------------------------------------------------------
    // Feature extraction
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
    // MLP
    // --------------------------------------------------------

    float probability = 0.0f;


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
    // Results
    // --------------------------------------------------------

    Serial.println();

    Serial.println(
        "RESAMPLED CYCLE"
    );

    Serial.println(
        "----------------------------------------"
    );


    Serial.printf(
        "Cycle length       : %d\n",
        TARGET_CYCLE_LEN
    );


    Serial.printf(
        "Resampling time    : %lu us\n",
        (unsigned long)(
            resample_end -
            resample_start
        )
    );


    Serial.println();

    Serial.println(
        "8 FEATURES"
    );

    Serial.println(
        "----------------------------------------"
    );


    Serial.printf(
        "crest              : %.8f\n",
        features[0]
    );


    Serial.printf(
        "skew               : %.8f\n",
        features[1]
    );


    Serial.printf(
        "kurt               : %.8f\n",
        features[2]
    );


    Serial.printf(
        "d1_ratio           : %.8f\n",
        features[3]
    );


    Serial.printf(
        "d2_ratio           : %.8f\n",
        features[4]
    );


    Serial.printf(
        "peak_asym          : %.8f\n",
        features[5]
    );


    Serial.printf(
        "energy_asym        : %.8f\n",
        features[6]
    );


    Serial.printf(
        "frac_small         : %.8f\n",
        features[7]
    );


    Serial.println();

    Serial.println(
        "MLP INFERENCE"
    );

    Serial.println(
        "----------------------------------------"
    );


    Serial.printf(
        "Arc probability    : %.8f\n",
        probability
    );


    Serial.printf(
        "Prediction         : %s\n",
        prediction ? "ARC" : "NORMAL"
    );


    Serial.println();

    Serial.println(
        "PROCESSING TIMING"
    );

    Serial.println(
        "----------------------------------------"
    );


    Serial.printf(
        "Slicer             : %lu us\n",
        (unsigned long)(
            slicer_end -
            slicer_start
        )
    );


    Serial.printf(
        "Resampling         : %lu us\n",
        (unsigned long)(
            resample_end -
            resample_start
        )
    );


    Serial.printf(
        "DSP                 : %lu us\n",
        (unsigned long)(
            dsp_end -
            dsp_start
        )
    );


    Serial.printf(
        "MLP                 : %lu us\n",
        (unsigned long)(
            mlp_end -
            mlp_start
        )
    );


    Serial.printf(
        "DSP + MLP           : %lu us\n",
        (unsigned long)(
            mlp_end -
            dsp_start
        )
    );


    Serial.println();

    Serial.println(
        "=================================================="
    );


    if (
        fabsf(
            first_frequency -
            50.0f
        ) <= 0.5f
    )
    {
        Serial.println(
            "STATUS: FIRST CYCLE NEAR 50 Hz"
        );
    }
    else
    {
        Serial.println(
            "STATUS: FIRST CYCLE NOT NEAR 50 Hz"
        );
    }


    Serial.println(
        "=================================================="
    );
}


// ------------------------------------------------------------
// LOOP
// ------------------------------------------------------------

void loop()
{
    delay(5000);
}