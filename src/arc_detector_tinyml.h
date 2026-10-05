#ifndef ARC_DETECTOR_TINYML_H
#define ARC_DETECTOR_TINYML_H

#include "dsp_feature_extractor.h"
#include "tinyml_engine.h"

// Master API: Processes raw ADC buffer and returns arc prediction
// Parameters:
//   adc_samples     - Pointer to array of float ADC current samples
//   num_samples     - Number of samples in buffer (e.g., 1000)
//   sample_rate_hz  - ADC sampling rate in Hz (e.g., 10000.0f)
//   out_probability - Pointer to store float arc probability output [0.0 - 1.0]
// Returns:
//   1 = Arc Fault Detected, 0 = Normal Operation
static inline int detect_arc_fault(const float *adc_samples, int num_samples, float sample_rate_hz, float *out_probability) {
    float features[14];
    extract_dsp_features(adc_samples, num_samples, sample_rate_hz, features);
    return tinyml_predict(features, out_probability);
}

#endif // ARC_DETECTOR_TINYML_H