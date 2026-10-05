#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dsp_feature_extractor.h"
#include "tinyml_engine.h"

#define BUFFER_SIZE 1000
#define SAMPLE_RATE 10000.0f

int main() {
    float raw_adc_buffer[BUFFER_SIZE];
    float extracted_features[14];
    float probability = 0.0f;

    printf("====================================================\n");
    printf("END-TO-END DSP + TINYML INFERENCE ENGINE TEST\n");
    printf("====================================================\n");

    // 1. Simulate Normal 50Hz Current Waveform (DC Offset: 2.98A, AC RMS: ~2.0A)
    for (int i = 0; i < BUFFER_SIZE; i++) {
        float t = (float)i / SAMPLE_RATE;
        float normal_signal = 2.98f + 2.83f * sinf(2.0f * M_PI * 50.0f * t);
        // Small background sensor noise
        float sensor_noise = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
        raw_adc_buffer[i] = normal_signal + sensor_noise;
    }

    extract_dsp_features(raw_adc_buffer, BUFFER_SIZE, SAMPLE_RATE, extracted_features);
    int pred_normal = tinyml_predict(extracted_features, &probability);
    printf("[TEST 1] Simulated Normal Waveform -> Result: %s (Prob: %.2f%%)\n",
           pred_normal == 1 ? "ARC FAULT" : "NORMAL", probability * 100.0f);

    // 2. Simulate Arc Fault (Normal Waveform + High-Frequency Arcing Spikes)
    for (int i = 0; i < BUFFER_SIZE; i++) {
        float t = (float)i / SAMPLE_RATE;
        float normal_signal = 2.98f + 2.83f * sinf(2.0f * M_PI * 50.0f * t);
        // Sharp high-frequency transient arcing spikes
        float arc_spike = 0.0f;
        if ((rand() % 10) < 3) {
            arc_spike = ((float)rand() / RAND_MAX - 0.5f) * 4.5f;
        }
        raw_adc_buffer[i] = normal_signal + arc_spike;
    }

    extract_dsp_features(raw_adc_buffer, BUFFER_SIZE, SAMPLE_RATE, extracted_features);
    int pred_arc = tinyml_predict(extracted_features, &probability);
    printf("[TEST 2] Simulated Arc Waveform   -> Result: %s (Prob: %.2f%%)\n",
           pred_arc == 1 ? "ARC FAULT" : "NORMAL", probability * 100.0f);

    printf("====================================================\n");
    return 0;
}