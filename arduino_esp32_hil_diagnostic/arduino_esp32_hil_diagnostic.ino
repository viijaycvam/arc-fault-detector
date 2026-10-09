#include <Arduino.h>
#include <math.h>

#include "driver/adc.h"
#include "esp_adc/adc_continuous.h"
#include "esp_err.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ============================================================
// ESP32 DAC -> ADC HIL DIAGNOSTIC
//
// Purpose:
//   Test only the DAC -> ADC -> DMA -> waveform path.
//
// No feature extraction.
// No neural network.
// No temporal voting.
//
// Safe connection:
//
//   GPIO25 / D25  --->  GPIO34 / D34
//
// NO MAINS.
// ============================================================


// ------------------------------------------------------------
// DAC
// ------------------------------------------------------------

#define DAC_PIN                 25

#define DAC_UPDATE_HZ           1000

#define DAC_WAVEFORM_HZ         50.0f

#define DAC_MIDPOINT            128.0f

#define DAC_AMPLITUDE           70.0f


// ------------------------------------------------------------
// ADC
// ------------------------------------------------------------

#define ADC_PIN                 34

#define ADC_CHANNEL             ADC_CHANNEL_6

#define SAMPLE_RATE_HZ          50000

#define ACQUISITION_SAMPLES     7040

#define FRAME_BYTES             256

#define DMA_STORE_BYTES         16384


// ------------------------------------------------------------
// Cycle detector
// ------------------------------------------------------------

#define LOWPASS_CUTOFF_HZ       100.0f

#define HYSTERESIS              0.10f

#define MIN_CROSSING_INTERVAL   700.0f

#define MAX_CROSSINGS           16


// ------------------------------------------------------------
// Diagnostic run
// ------------------------------------------------------------

#define DIAGNOSTIC_WINDOWS      20


// ------------------------------------------------------------
// Buffers
// ------------------------------------------------------------

static uint16_t adc_samples[
    ACQUISITION_SAMPLES
];

static float crossing_positions[
    MAX_CROSSINGS
];


// ------------------------------------------------------------
// ADC handle
// ------------------------------------------------------------

static adc_continuous_handle_t adc_handle = NULL;


// ------------------------------------------------------------
// DAC task
// ------------------------------------------------------------

static volatile bool waveform_running = false;

static TaskHandle_t dac_task_handle = NULL;


// ============================================================
// DAC waveform generator
//
// Uses vTaskDelayUntil rather than repeatedly calling
// vTaskDelay(1), reducing cumulative scheduler drift.
// ============================================================

static void dac_waveform_task(
    void *parameter
)
{
    (void)parameter;


    uint32_t sample_index = 0;


    const float phase_step =
        2.0f *
        (float)M_PI *
        DAC_WAVEFORM_HZ /
        (float)DAC_UPDATE_HZ;


    TickType_t last_wake =
        xTaskGetTickCount();


    const TickType_t period_ticks =
        pdMS_TO_TICKS(1);


    dacWrite(
        DAC_PIN,
        (uint8_t)DAC_MIDPOINT
    );


    while (
        waveform_running
    )
    {
        float phase =
            phase_step *
            (float)sample_index;


        float sine_value =
            sinf(
                phase
            );


        float dac_value =
            DAC_MIDPOINT +
            DAC_AMPLITUDE *
            sine_value;


        if (
            dac_value <
            0.0f
        )
        {
            dac_value = 0.0f;
        }


        if (
            dac_value >
            255.0f
        )
        {
            dac_value = 255.0f;
        }


        dacWrite(
            DAC_PIN,
            (uint8_t)dac_value
        );


        sample_index++;


        vTaskDelayUntil(
            &last_wake,
            period_ticks
        );
    }


    dacWrite(
        DAC_PIN,
        (uint8_t)DAC_MIDPOINT
    );


    dac_task_handle = NULL;


    vTaskDelete(NULL);
}


// ============================================================
// Start DAC
// ============================================================

static bool start_waveform()
{
    waveform_running = true;


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
        waveform_running = false;

        dac_task_handle = NULL;

        return false;
    }


    return true;
}


// ============================================================
// Stop DAC
// ============================================================

static void stop_waveform()
{
    waveform_running = false;


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


// ============================================================
// Start ADC
// ============================================================

static bool start_adc()
{
    adc_continuous_handle_cfg_t handle_config = {};


    handle_config.max_store_buf_size =
        DMA_STORE_BYTES;


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
            "ADC HANDLE ERROR: 0x%X\n",
            err
        );


        adc_handle = NULL;


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
            "ADC CONFIG ERROR: 0x%X\n",
            err
        );


        adc_continuous_deinit(
            adc_handle
        );


        adc_handle = NULL;


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
            "ADC START ERROR: 0x%X\n",
            err
        );


        adc_continuous_deinit(
            adc_handle
        );


        adc_handle = NULL;


        return false;
    }


    return true;
}


// ============================================================
// Stop ADC
// ============================================================

static void stop_adc()
{
    if (
        adc_handle != NULL
    )
    {
        adc_continuous_stop(
            adc_handle
        );


        adc_continuous_deinit(
            adc_handle
        );


        adc_handle = NULL;
    }
}


// ============================================================
// Flush initial DMA data
// ============================================================

static void flush_startup_data()
{
    uint8_t dma_buffer[
        FRAME_BYTES
    ];


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


// ============================================================
// Acquire 7040 ADC samples
// ============================================================

static bool acquire_samples(
    uint32_t &elapsed_us
)
{
    uint8_t dma_buffer[
        FRAME_BYTES
    ];


    uint32_t sample_count = 0;


    uint32_t start_us =
        micros();


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
            elapsed_us =
                micros() -
                start_us;


            return false;
        }


        for (
            uint32_t i = 0;
            i + 1 < bytes_read;
            i += 2
        )
        {
            if (
                sample_count >=
                ACQUISITION_SAMPLES
            )
            {
                break;
            }


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
                    word >>
                    12
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


// ============================================================
// Mean
// ============================================================

static float calculate_mean()
{
    uint64_t sum = 0;


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


// ============================================================
// Minimum / maximum / peak
// ============================================================

static void calculate_stats(
    uint16_t &min_value,
    uint16_t &max_value,
    float dc_mean,
    float &peak_centered
)
{
    min_value = 4095;

    max_value = 0;

    peak_centered = 0.0f;


    for (
        int i = 0;
        i < ACQUISITION_SAMPLES;
        i++
    )
    {
        uint16_t value =
            adc_samples[i];


        if (
            value <
            min_value
        )
        {
            min_value =
                value;
        }


        if (
            value >
            max_value
        )
        {
            max_value =
                value;
        }


        float centered =
            fabsf(
                (float)value -
                dc_mean
            );


        if (
            centered >
            peak_centered
        )
        {
            peak_centered =
                centered;
        }
    }
}


// ============================================================
// Robust zero crossing detector
// ============================================================

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
                centered -
                previous_filtered
            );


        float previous =
            previous_filtered;


        float current =
            current_filtered;


        if (
            state ==
            0
        )
        {
            if (
                current <=
                -HYSTERESIS
            )
            {
                state = 1;
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


                    state = 0;
                }
            }
        }


        previous_filtered =
            current_filtered;
    }


    return crossing_count;
}


// ============================================================
// Print frequency information
// ============================================================

static void print_frequency_info(
    int crossing_count
)
{
    if (
        crossing_count <
        2
    )
    {
        Serial.println(
            "Frequency: unavailable"
        );


        return;
    }


    int usable_cycles =
        crossing_count - 1;


    if (
        usable_cycles >
        5
    )
    {
        usable_cycles = 5;
    }


    for (
        int i = 0;
        i < usable_cycles;
        i++
    )
    {
        float period =
            crossing_positions[i + 1]
            -
            crossing_positions[i];


        if (
            period <=
            0.0f
        )
        {
            continue;
        }


        float frequency =
            (float)SAMPLE_RATE_HZ /
            period;


        Serial.printf(
            "C%d frequency: %.3f Hz\n",
            i + 1,
            frequency
        );
    }
}


// ============================================================
// Main diagnostic test
// ============================================================

static void run_diagnostic()
{
    uint32_t valid_windows = 0;

    uint32_t invalid_windows = 0;

    uint32_t failed_windows = 0;


    Serial.println();


    Serial.println(
        "=================================================="
    );


    Serial.println(
        "HIL DAC -> ADC DIAGNOSTIC"
    );


    Serial.println(
        "=================================================="
    );


    for (
        uint32_t window = 1;
        window <= DIAGNOSTIC_WINDOWS;
        window++
    )
    {
        // ----------------------------------------------------
        // Start a fresh ADC session.
        // ----------------------------------------------------

        if (
            !start_adc()
        )
        {
            failed_windows++;


            Serial.printf(
                "WINDOW %lu: ADC START FAILED\n",
                (unsigned long)window
            );


            delay(50);


            continue;
        }


        flush_startup_data();


        // ----------------------------------------------------
        // Acquire one complete window.
        // ----------------------------------------------------

        uint32_t acquisition_us = 0;


        bool acquired =
            acquire_samples(
                acquisition_us
            );


        stop_adc();


        if (
            !acquired
        )
        {
            failed_windows++;


            Serial.printf(
                "WINDOW %lu: ADC ACQUISITION FAILED\n",
                (unsigned long)window
            );


            delay(50);


            continue;
        }


        // ----------------------------------------------------
        // Statistics.
        // ----------------------------------------------------

        float dc_mean =
            calculate_mean();


        float peak_centered = 0.0f;


        uint16_t min_value = 4095;

        uint16_t max_value = 0;


        calculate_stats(
            min_value,
            max_value,
            dc_mean,
            peak_centered
        );


        // ----------------------------------------------------
        // Crossing detection.
        // ----------------------------------------------------

        uint32_t slicer_start =
            micros();


        int crossing_count =
            detect_crossings(
                dc_mean,
                peak_centered
            );


        uint32_t slicer_end =
            micros();


        float observed_rate =
            (
                (
                    (float)
                    ACQUISITION_SAMPLES
                    /
                    (float)
                    acquisition_us
                )
                *
                1000000.0f
            );


        // ----------------------------------------------------
        // Print window.
        // ----------------------------------------------------

        Serial.println();


        Serial.println(
            "--------------------------------------------------"
        );


        Serial.printf(
            "WINDOW %lu\n",
            (unsigned long)window
        );


        Serial.printf(
            "Acquisition: %.3f ms\n",
            acquisition_us /
            1000.0f
        );


        Serial.printf(
            "Observed rate: %.2f samples/sec\n",
            observed_rate
        );


        Serial.printf(
            "ADC min/max: %u / %u\n",
            min_value,
            max_value
        );


        Serial.printf(
            "ADC mean: %.2f\n",
            dc_mean
        );


        Serial.printf(
            "ADC centered peak: %.2f\n",
            peak_centered
        );


        Serial.printf(
            "Crossings: %d\n",
            crossing_count
        );


        Serial.printf(
            "Slicer time: %lu us\n",
            (unsigned long)(
                slicer_end -
                slicer_start
            )
        );


        print_frequency_info(
            crossing_count
        );


        // ----------------------------------------------------
        // Window classification.
        //
        // For this diagnostic, >= 6 crossings means the
        // waveform path produced enough cycles for the
        // normal detector.
        // ----------------------------------------------------

        if (
            crossing_count >=
            6
        )
        {
            valid_windows++;


            Serial.println(
                "WINDOW STATUS: PASS"
            );
        }
        else
        {
            invalid_windows++;


            Serial.println(
                "WINDOW STATUS: FAIL"
            );
        }


        delay(10);
    }


    // --------------------------------------------------------
    // Final diagnostic summary
    // --------------------------------------------------------

    Serial.println();


    Serial.println(
        "=================================================="
    );


    Serial.println(
        "DIAGNOSTIC SUMMARY"
    );


    Serial.println(
        "=================================================="
    );


    Serial.printf(
        "Total windows: %d\n",
        DIAGNOSTIC_WINDOWS
    );


    Serial.printf(
        "Valid windows: %lu\n",
        (unsigned long)valid_windows
    );


    Serial.printf(
        "Invalid windows: %lu\n",
        (unsigned long)invalid_windows
    );


    Serial.printf(
        "ADC failures: %lu\n",
        (unsigned long)failed_windows
    );


    if (
        invalid_windows == 0 &&
        failed_windows == 0
    )
    {
        Serial.println(
            "FINAL DIAGNOSTIC: PASS"
        );
    }
    else
    {
        Serial.println(
            "FINAL DIAGNOSTIC: INVESTIGATE HIL PATH"
        );
    }


    Serial.println(
        "=================================================="
    );
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(
        115200
    );


    delay(1000);


    Serial.println();


    Serial.println(
        "ESP32 HIL DIAGNOSTIC"
    );


    Serial.println(
        "GPIO25 DAC -> GPIO34 ADC"
    );


    Serial.println(
        "NO MAINS"
    );


    Serial.printf(
        "DAC waveform: %.1f Hz\n",
        DAC_WAVEFORM_HZ
    );


    Serial.printf(
        "DAC update: %d Hz\n",
        DAC_UPDATE_HZ
    );


    Serial.printf(
        "ADC rate: %d Hz\n",
        SAMPLE_RATE_HZ
    );


    Serial.printf(
        "Samples/window: %d\n",
        ACQUISITION_SAMPLES
    );


    Serial.printf(
        "Diagnostic windows: %d\n",
        DIAGNOSTIC_WINDOWS
    );


    // Start the safe DAC source once.
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


    delay(500);


    run_diagnostic();


    stop_waveform();


    Serial.println();


    Serial.println(
        "DIAGNOSTIC COMPLETE."
    );
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
    delay(1000);
}