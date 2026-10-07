#include <Arduino.h>

#include "driver/adc.h"
#include "esp_adc/adc_continuous.h"
#include "esp_err.h"
#include "soc/soc_caps.h"

// ============================================================
// ESP32 ADC DIRECT DMA THROUGHPUT TEST
//
// Board : DOIT ESP32 DEVKIT V1 / classic ESP32
// ADC   : GPIO34 = ADC1_CH6
//
// IMPORTANT:
//   GPIO34 must be UNCONNECTED.
//
// Test:
//   Configure ADC for 50 kHz.
//   Flush startup DMA data.
//   Count actual ADC results for 10 seconds.
//
// This avoids using GPIO25 or waveform edge detection.
// ============================================================

#define ADC_PIN             34
#define ADC_CHANNEL         ADC_CHANNEL_6

#define ADC_RATE_HZ         50000

#define TEST_DURATION_MS    60000

#define FRAME_BYTES         256

volatile uint32_t overflow_count = 0;

adc_continuous_handle_t adc_handle = NULL;

// ------------------------------------------------------------
// ADC pool overflow callback
// ------------------------------------------------------------

bool IRAM_ATTR on_pool_overflow(
    adc_continuous_handle_t handle,
    const adc_continuous_evt_data_t *edata,
    void *user_data
)
{
    overflow_count++;
    return false;
}

// ------------------------------------------------------------
// Start ADC
// ------------------------------------------------------------

bool start_adc()
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
            "ERROR new_handle: 0x%X\n",
            err
        );

        adc_handle = NULL;
        return false;
    }

    // --------------------------------------------------------
    // Configure GPIO34 / ADC1_CH6
    // --------------------------------------------------------

    adc_digi_pattern_config_t pattern = {};

    pattern.atten = ADC_ATTEN_DB_11;
    pattern.channel = ADC_CHANNEL;
    pattern.unit = ADC_UNIT_1;
    pattern.bit_width = ADC_BITWIDTH_12;

    adc_continuous_config_t config = {};

    config.sample_freq_hz = ADC_RATE_HZ;
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
            "ERROR adc_config: 0x%X\n",
            err
        );

        adc_continuous_deinit(adc_handle);
        adc_handle = NULL;

        return false;
    }

    // --------------------------------------------------------
    // Register overflow callback
    // --------------------------------------------------------

    adc_continuous_evt_cbs_t callbacks = {};

    callbacks.on_pool_ovf =
        on_pool_overflow;

    err =
        adc_continuous_register_event_callbacks(
            adc_handle,
            &callbacks,
            NULL
        );

    if (err != ESP_OK)
    {
        Serial.printf(
            "ERROR register callbacks: 0x%X\n",
            err
        );

        adc_continuous_deinit(adc_handle);
        adc_handle = NULL;

        return false;
    }

    // --------------------------------------------------------
    // Start
    // --------------------------------------------------------

    err =
        adc_continuous_start(
            adc_handle
        );

    if (err != ESP_OK)
    {
        Serial.printf(
            "ERROR adc_start: 0x%X\n",
            err
        );

        adc_continuous_deinit(adc_handle);
        adc_handle = NULL;

        return false;
    }

    return true;
}

// ------------------------------------------------------------
// Flush all already-buffered startup data.
//
// We intentionally throw this data away so that the timing
// measurement begins only when the DMA ring buffer is empty.
// ------------------------------------------------------------

void flush_startup_data()
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
                "ERROR during flush: 0x%X\n",
                err
            );

            break;
        }

        // Data intentionally discarded.
    }
}

// ------------------------------------------------------------
// Run the 60-second throughput measurement
// ------------------------------------------------------------

bool measure_rate(
    uint64_t &sample_count,
    uint32_t &elapsed_ms
)
{
    uint8_t dma_buffer[FRAME_BYTES];

    sample_count = 0;

    // Ensure overflow counter starts at zero.
    overflow_count = 0;

    // --------------------------------------------------------
    // First successful frame after flush.
    //
    // We do not start timing until the startup buffer has
    // been drained.
    // --------------------------------------------------------

    uint32_t bytes_read = 0;

    esp_err_t err =
        adc_continuous_read(
            adc_handle,
            dma_buffer,
            sizeof(dma_buffer),
            &bytes_read,
            100
        );

    if (err != ESP_OK)
    {
        Serial.printf(
            "ERROR first read: 0x%X\n",
            err
        );

        return false;
    }

    // Discard this frame too.
    //
    // This guarantees the timing interval starts with a
    // completely fresh DMA stream.
    // --------------------------------------------------------

    uint32_t start_ms = millis();

    while (
        (millis() - start_ms) <
        TEST_DURATION_MS
    )
    {
        bytes_read = 0;

        err =
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
                "ERROR ADC read: 0x%X\n",
                err
            );

            return false;
        }

        // ----------------------------------------------------
        // On the classic ESP32:
        //
        // SOC_ADC_DIGI_RESULT_BYTES = 2
        //
        // ESP-IDF's continuous ADC examples count logical
        // results using ret_num / SOC_ADC_DIGI_RESULT_BYTES.
        // ----------------------------------------------------

        sample_count +=
            bytes_read /
            SOC_ADC_DIGI_RESULT_BYTES;
    }

    elapsed_ms =
        millis() - start_ms;

    return true;
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
        "========================================"
    );

    Serial.println(
        "ESP32 ADC DIRECT DMA RATE TEST"
    );

    Serial.println(
        "========================================"
    );

    Serial.printf(
        "CPU frequency       : %d MHz\n",
        getCpuFrequencyMhz()
    );

    Serial.printf(
        "ESP32 Arduino core  : %s\n",
        ESP_ARDUINO_VERSION_STR
    );

    Serial.printf(
        "ESP-IDF version     : %s\n",
        esp_get_idf_version()
    );

    Serial.printf(
        "ADC pin             : GPIO%d\n",
        ADC_PIN
    );

    Serial.println(
        "ADC channel         : ADC1_CH6"
    );

    Serial.printf(
        "Configured rate     : %d samples/sec\n",
        ADC_RATE_HZ
    );

    Serial.printf(
        "Measurement time    : %d ms\n",
        TEST_DURATION_MS
    );

    Serial.println();
    Serial.println(
        "GPIO34 MUST BE UNCONNECTED."
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
            delay(1000);
    }

    Serial.println();
    Serial.println(
        "ADC DMA started."
    );

    // --------------------------------------------------------
    // Flush startup frames
    // --------------------------------------------------------

    Serial.println(
        "Flushing startup DMA data..."
    );

    flush_startup_data();

    Serial.println(
        "Startup DMA data flushed."
    );

    // --------------------------------------------------------
    // Measure
    // --------------------------------------------------------

    Serial.println();
    Serial.println(
        "Starting 60-second measurement..."
    );

    uint64_t sample_count = 0;
    uint32_t elapsed_ms = 0;

    bool ok =
        measure_rate(
            sample_count,
            elapsed_ms
        );

    // --------------------------------------------------------
    // Stop
    // --------------------------------------------------------

    adc_continuous_stop(
        adc_handle
    );

    adc_continuous_deinit(
        adc_handle
    );

    adc_handle = NULL;

    if (!ok)
    {
        Serial.println();
        Serial.println(
            "STATUS: RATE MEASUREMENT FAILED"
        );

        while (true)
            delay(1000);
    }

    // --------------------------------------------------------
    // Calculate measured rate
    // --------------------------------------------------------

    double measured_rate = 0.0;

    if (elapsed_ms > 0)
    {
        measured_rate =
            (
                (double)sample_count *
                1000.0
            ) /
            (double)elapsed_ms;
    }

    double rate_error =
        100.0 *
        (
            measured_rate -
            (double)ADC_RATE_HZ
        ) /
        (double)ADC_RATE_HZ;

    double expected_samples =
        (
            (double)ADC_RATE_HZ *
            (double)TEST_DURATION_MS
        ) /
        1000.0;

    double sample_count_error =
        100.0 *
        (
            (double)sample_count -
            expected_samples
        ) /
        expected_samples;

    // --------------------------------------------------------
    // Results
    // --------------------------------------------------------

    Serial.println();
    Serial.println(
        "========================================"
    );

    Serial.println(
        "60-SECOND ADC RATE RESULT"
    );

    Serial.println(
        "========================================"
    );

    Serial.printf(
        "Configured rate      : %d Hz\n",
        ADC_RATE_HZ
    );

    Serial.printf(
        "Expected samples     : %.0f\n",
        expected_samples
    );

    Serial.printf(
        "Measured samples     : %llu\n",
        (unsigned long long)sample_count
    );

    Serial.printf(
        "Elapsed time         : %lu ms\n",
        (unsigned long)elapsed_ms
    );

    Serial.printf(
        "Measured rate        : %.2f Hz\n",
        measured_rate
    );

    Serial.printf(
        "Rate error           : %+.3f %%\n",
        rate_error
    );

    Serial.printf(
        "Sample-count error   : %+.3f %%\n",
        sample_count_error
    );

    Serial.printf(
        "DMA overflow count   : %lu\n",
        (unsigned long)overflow_count
    );

    Serial.println();

    if (
        fabs(rate_error) <= 1.0 &&
        overflow_count == 0
    )
    {
        Serial.println(
            "STATUS: SAMPLE RATE VERIFIED"
        );

        Serial.println(
            "STATUS: NO DMA OVERFLOW"
        );
    }
    else
    {
        Serial.println(
            "STATUS: SAMPLE RATE NOT VERIFIED"
        );

        if (overflow_count != 0)
        {
            Serial.println(
                "WARNING: DMA OVERFLOW DETECTED"
            );
        }
    }

    Serial.println(
        "========================================"
    );
}

void loop()
{
    delay(1000);
}