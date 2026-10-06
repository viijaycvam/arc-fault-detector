#include <Arduino.h>

#include "driver/adc.h"
#include "esp_adc/adc_continuous.h"
#include "esp_err.h"
#include "soc/soc_caps.h"

// ============================================================
// ESP32 ADC DMA - TIMING-VALIDATED 1000 SAMPLE TEST
//
// Board : DOIT ESP32 DEVKIT V1 / classic ESP32
// Pin   : GPIO34 = ADC1_CH6
// Rate  : 50 kHz
// Data  : 1000 raw ADC samples
//
// GPIO34 MUST REMAIN UNCONNECTED.
// ============================================================

#define ADC_PIN         34
#define ADC_CHANNEL     ADC_CHANNEL_6

#define SAMPLE_RATE_HZ  50000
#define NUM_SAMPLES     1000
#define FRAME_BYTES     256

uint16_t samples[NUM_SAMPLES];

adc_continuous_handle_t adc_handle = NULL;

void setup()
{
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("========================================");
    Serial.println("ESP32 ADC DMA TIMING TEST");
    Serial.println("========================================");

    Serial.printf("ADC pin        : GPIO%d\n", ADC_PIN);
    Serial.println("ADC channel    : ADC1_CH6");
    Serial.printf("Sample rate    : %d Hz\n", SAMPLE_RATE_HZ);
    Serial.printf("Samples        : %d\n", NUM_SAMPLES);
    Serial.printf("Target window  : %.2f ms\n",
                  1000.0f * NUM_SAMPLES / SAMPLE_RATE_HZ);

    // --------------------------------------------------------
    // Create ADC DMA handle
    // --------------------------------------------------------

    adc_continuous_handle_cfg_t handle_config = {};

    handle_config.max_store_buf_size = 4096;
    handle_config.conv_frame_size = FRAME_BYTES;

    esp_err_t err = adc_continuous_new_handle(
        &handle_config,
        &adc_handle
    );

    if (err != ESP_OK)
    {
        Serial.printf(
            "ERROR: new handle failed: 0x%X\n",
            err
        );

        while (true)
            delay(1000);
    }

    // --------------------------------------------------------
    // Configure ADC1 CH6 / GPIO34
    // --------------------------------------------------------

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

    err = adc_continuous_config(
        adc_handle,
        &config
    );

    if (err != ESP_OK)
    {
        Serial.printf(
            "ERROR: ADC config failed: 0x%X\n",
            err
        );

        adc_continuous_deinit(adc_handle);
        adc_handle = NULL;

        while (true)
            delay(1000);
    }

    // --------------------------------------------------------
    // IMPORTANT:
    // Start timing BEFORE ADC starts.
    // No Serial printing occurs while acquisition is underway.
    // --------------------------------------------------------

    uint32_t start_us = micros();

    err = adc_continuous_start(adc_handle);

    if (err != ESP_OK)
    {
        Serial.printf(
            "ERROR: ADC start failed: 0x%X\n",
            err
        );

        adc_continuous_deinit(adc_handle);
        adc_handle = NULL;

        while (true)
            delay(1000);
    }

    uint8_t dma_buffer[FRAME_BYTES];

    uint32_t sample_count = 0;
    uint32_t dma_frames = 0;

    // --------------------------------------------------------
    // Collect exactly 1000 valid ADC samples
    // --------------------------------------------------------

    while (sample_count < NUM_SAMPLES)
    {
        uint32_t bytes_read = 0;

        err = adc_continuous_read(
            adc_handle,
            dma_buffer,
            sizeof(dma_buffer),
            &bytes_read,
            100
        );

        if (err != ESP_OK)
        {
            Serial.printf(
                "ERROR: ADC read failed: 0x%X\n",
                err
            );

            adc_continuous_stop(adc_handle);
            adc_continuous_deinit(adc_handle);
            adc_handle = NULL;

            while (true)
                delay(1000);
        }

        dma_frames++;

        for (
            uint32_t i = 0;
            i + 1 < bytes_read && sample_count < NUM_SAMPLES;
            i += 2
        )
        {
            uint16_t word =
                (uint16_t)dma_buffer[i] |
                ((uint16_t)dma_buffer[i + 1] << 8);

            uint16_t channel =
                (word >> 12) & 0x0F;

            uint16_t raw =
                word & 0x0FFF;

            if (channel == ADC_CHANNEL)
            {
                samples[sample_count] = raw;
                sample_count++;
            }
        }
    }

    uint32_t end_us = micros();

    adc_continuous_stop(adc_handle);
    adc_continuous_deinit(adc_handle);
    adc_handle = NULL;

    // --------------------------------------------------------
    // Statistics
    // --------------------------------------------------------

    uint16_t min_value = 4095;
    uint16_t max_value = 0;

    uint64_t sum = 0;

    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        if (samples[i] < min_value)
            min_value = samples[i];

        if (samples[i] > max_value)
            max_value = samples[i];

        sum += samples[i];
    }

    uint32_t elapsed_us = end_us - start_us;

    float elapsed_ms =
        elapsed_us / 1000.0f;

    float effective_rate =
        (NUM_SAMPLES / (float)elapsed_us) * 1000000.0f;

    float mean =
        (float)sum / NUM_SAMPLES;

    // --------------------------------------------------------
    // Results
    // --------------------------------------------------------

    Serial.println();
    Serial.println("========================================");
    Serial.println("ADC DMA TIMING RESULT");
    Serial.println("========================================");

    Serial.printf(
        "Samples collected : %lu / %d\n",
        (unsigned long)sample_count,
        NUM_SAMPLES
    );

    Serial.printf(
        "DMA frames used   : %lu\n",
        (unsigned long)dma_frames
    );

    Serial.printf(
        "Elapsed time      : %.3f ms\n",
        elapsed_ms
    );

    Serial.printf(
        "Measured rate     : %.1f samples/sec\n",
        effective_rate
    );

    Serial.printf(
        "Configured rate   : %d samples/sec\n",
        SAMPLE_RATE_HZ
    );

    Serial.printf(
        "Min raw ADC       : %u\n",
        min_value
    );

    Serial.printf(
        "Max raw ADC       : %u\n",
        max_value
    );

    Serial.printf(
        "Mean raw ADC      : %.2f\n",
        mean
    );

    Serial.println();
    Serial.println("First 20 raw ADC samples:");

    for (int i = 0; i < 20; i++)
    {
        Serial.printf(
            "%4d : %u\n",
            i,
            samples[i]
        );
    }

    Serial.println();
    Serial.println("========================================");

    if (
        sample_count == NUM_SAMPLES &&
        elapsed_ms >= 18.0f &&
        elapsed_ms <= 23.0f
    )
    {
        Serial.println(
            "STATUS: 50 kHz ADC ACQUISITION VERIFIED"
        );
    }
    else
    {
        Serial.println(
            "STATUS: TIMING STILL NEEDS INVESTIGATION"
        );
    }

    Serial.println("========================================");
}

void loop()
{
    // Runs once.
}