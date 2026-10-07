#include <Arduino.h>

#include "driver/adc.h"
#include "esp_adc/adc_continuous.h"
#include "esp_err.h"
#include "soc/soc_caps.h"

// ============================================================
// ESP32 DC STABILITY / NOISE TEST
//
// DAC:
//   GPIO25 = DAC1
//
// ADC:
//   GPIO34 = ADC1_CH6
//
// Connection:
//   D25 -> D34
//
// DAC value:
//   100 / 255 * 3.3 V ≈ 1.29 V nominal
//
// This measures the combined stability of:
//   ESP32 DAC -> ADC -> DMA
//
// It is NOT the final intrinsic ADC noise-floor test.
// The true 1.3 V external DC test will be done later.
// ============================================================

#define DAC_PIN             25
#define DAC_VALUE           100

#define ADC_PIN             34
#define ADC_CHANNEL         ADC_CHANNEL_6

#define ADC_RATE_HZ         50000
#define NUM_SAMPLES         1000
#define FRAME_BYTES         256

uint16_t samples[NUM_SAMPLES];

adc_continuous_handle_t adc_handle = NULL;

// ------------------------------------------------------------
// Start ADC DMA
// ------------------------------------------------------------

bool start_adc()
{
    adc_continuous_handle_cfg_t handle_config = {};

    handle_config.max_store_buf_size = 4096;
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

        return false;
    }

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

    if (err != ESP_OK)
    {
        Serial.printf(
            "ERROR adc_start: 0x%X\n",
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

// ------------------------------------------------------------
// Stop ADC
// ------------------------------------------------------------

void stop_adc()
{
    if (adc_handle != NULL)
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

// ------------------------------------------------------------
// Flush initial DMA samples
// ------------------------------------------------------------

void flush_adc()
{
    uint8_t buffer[FRAME_BYTES];

    while (true)
    {
        uint32_t bytes_read = 0;

        esp_err_t err =
            adc_continuous_read(
                adc_handle,
                buffer,
                sizeof(buffer),
                &bytes_read,
                0
            );

        if (err == ESP_ERR_TIMEOUT)
            break;

        if (err != ESP_OK)
            break;
    }
}

// ------------------------------------------------------------
// Capture exactly 1000 ADC samples
// ------------------------------------------------------------

bool capture_samples()
{
    uint8_t buffer[FRAME_BYTES];

    uint32_t sample_count = 0;

    while (
        sample_count < NUM_SAMPLES
    )
    {
        uint32_t bytes_read = 0;

        esp_err_t err =
            adc_continuous_read(
                adc_handle,
                buffer,
                sizeof(buffer),
                &bytes_read,
                100
            );

        if (err != ESP_OK)
        {
            Serial.printf(
                "ERROR adc_read: 0x%X\n",
                err
            );

            return false;
        }

        for (
            uint32_t i = 0;
            i + 1 < bytes_read &&
            sample_count < NUM_SAMPLES;
            i += SOC_ADC_DIGI_RESULT_BYTES
        )
        {
            adc_digi_output_data_t *result =
                (adc_digi_output_data_t *)&buffer[i];

            uint16_t channel =
                result->type1.channel;

            uint16_t raw =
                result->type1.data;

            if (
                channel ==
                ADC_CHANNEL
            )
            {
                samples[sample_count] =
                    raw;

                sample_count++;
            }
        }
    }

    return true;
}

// ------------------------------------------------------------
// Print statistics
// ------------------------------------------------------------

void print_statistics()
{
    uint16_t min_value = 4095;
    uint16_t max_value = 0;

    uint64_t sum = 0;

    for (
        int i = 0;
        i < NUM_SAMPLES;
        i++
    )
    {
        if (
            samples[i] <
            min_value
        )
        {
            min_value =
                samples[i];
        }

        if (
            samples[i] >
            max_value
        )
        {
            max_value =
                samples[i];
        }

        sum +=
            samples[i];
    }

    float mean =
        (float)sum /
        NUM_SAMPLES;

    float variance =
        0.0f;

    for (
        int i = 0;
        i < NUM_SAMPLES;
        i++
    )
    {
        float d =
            (float)samples[i] -
            mean;

        variance +=
            d * d;
    }

    variance /=
        NUM_SAMPLES;

    float stddev =
        sqrtf(variance);

    Serial.println();
    Serial.println(
        "========================================"
    );

    Serial.println(
        "DC STABILITY RESULT"
    );

    Serial.println(
        "========================================"
    );

    Serial.printf(
        "DAC pin             : GPIO%d\n",
        DAC_PIN
    );

    Serial.printf(
        "DAC code            : %d / 255\n",
        DAC_VALUE
    );

    Serial.printf(
        "Nominal DAC voltage : %.3f V\n",
        3.3f *
        DAC_VALUE /
        255.0f
    );

    Serial.printf(
        "ADC samples         : %d\n",
        NUM_SAMPLES
    );

    Serial.printf(
        "Min ADC             : %u\n",
        min_value
    );

    Serial.printf(
        "Max ADC             : %u\n",
        max_value
    );

    Serial.printf(
        "Mean ADC            : %.2f\n",
        mean
    );

    Serial.printf(
        "Std dev             : %.4f counts\n",
        stddev
    );

    Serial.printf(
        "Peak-to-peak        : %u counts\n",
        max_value - min_value
    );
}

// ------------------------------------------------------------
// Print first 1000 samples as CSV
// ------------------------------------------------------------

void print_csv()
{
    Serial.println();
    Serial.println(
        "BEGIN_CAPTURE_CSV"
    );

    for (
        int i = 0;
        i < NUM_SAMPLES;
        i++
    )
    {
        Serial.print(
            samples[i]
        );

        if (
            i <
            NUM_SAMPLES - 1
        )
        {
            Serial.print(",");
        }
    }

    Serial.println();

    Serial.println(
        "END_CAPTURE_CSV"
    );
}

// ------------------------------------------------------------
// Setup
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
        "ESP32 DC STABILITY TEST"
    );

    Serial.println(
        "INTERNAL DAC -> ADC DMA"
    );

    Serial.println(
        "========================================"
    );

    Serial.printf(
        "CPU frequency    : %d MHz\n",
        getCpuFrequencyMhz()
    );

    Serial.printf(
        "Arduino core     : %s\n",
        ESP_ARDUINO_VERSION_STR
    );

    Serial.printf(
        "ESP-IDF          : %s\n",
        esp_get_idf_version()
    );

    Serial.printf(
        "DAC pin          : GPIO%d\n",
        DAC_PIN
    );

    Serial.printf(
        "DAC value        : %d / 255\n",
        DAC_VALUE
    );

    Serial.printf(
        "ADC pin          : GPIO%d\n",
        ADC_PIN
    );

    Serial.printf(
        "ADC rate         : %d Hz\n",
        ADC_RATE_HZ
    );

    Serial.printf(
        "Samples          : %d\n",
        NUM_SAMPLES
    );

    Serial.println();
    Serial.println(
        "WIRING:"
    );

    Serial.println(
        "D25 -> D34"
    );

    // --------------------------------------------------------
    // Set DAC to nominal 1.29 V.
    // --------------------------------------------------------

    dacWrite(
        DAC_PIN,
        DAC_VALUE
    );

    Serial.printf(
        "DAC set to nominal %.3f V\n",
        3.3f *
        DAC_VALUE /
        255.0f
    );

    // Allow DAC output to settle.
    delay(100);

    // --------------------------------------------------------
    // Start ADC
    // --------------------------------------------------------

    if (!start_adc())
    {
        Serial.println(
            "STATUS: ADC START FAILED"
        );

        while (true)
            delay(1000);
    }

    Serial.println(
        "ADC DMA started."
    );

    // Discard data accumulated during settling.
    flush_adc();

    Serial.println(
        "Startup DMA data flushed."
    );

    // --------------------------------------------------------
    // Capture
    // --------------------------------------------------------

    Serial.println();
    Serial.println(
        "Capturing 1000 DC samples..."
    );

    bool ok =
        capture_samples();

    stop_adc();

    dacDisable(
        DAC_PIN
    );

    if (!ok)
    {
        Serial.println(
            "STATUS: CAPTURE FAILED"
        );

        while (true)
            delay(1000);
    }

    Serial.println();
    Serial.println(
        "1000 samples captured."
    );

    print_statistics();

    print_csv();

    Serial.println();
    Serial.println(
        "========================================"
    );

    Serial.println(
        "STATUS: DC STABILITY TEST COMPLETE"
    );

    Serial.println(
        "========================================"
    );
}

// ------------------------------------------------------------
// Loop
// ------------------------------------------------------------

void loop()
{
    delay(1000);
}