#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#define CYCLE_LEN 1000

void extract_features_8(const float raw_samples[CYCLE_LEN], float features_out[8]);

#endif // FEATURE_EXTRACTION_H