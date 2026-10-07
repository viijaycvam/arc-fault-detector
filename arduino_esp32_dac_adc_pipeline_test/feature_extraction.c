#include <math.h>
#include <stdlib.h>
#include <float.h>
#include "feature_extraction.h"

/*
 * Option-B 8-feature time-domain extractor.
 *
 * This implementation is designed to match src\20_features_b.py:
 *
 * Python feature order:
 *   0 = crest
 *   1 = skew
 *   2 = kurt
 *   3 = d1_ratio
 *   4 = d2_ratio
 *   5 = peak_asym
 *   6 = energy_asym
 *   7 = frac_small
 *
 * Input:
 *   raw_samples[1000]
 *
 * Output:
 *   features_out[8]
 */

#define NUM_SAMPLES 1000

void extract_features_8(
    const float raw_samples[NUM_SAMPLES],
    float features_out[8]
)
{
    /* =========================================================
       PASS 1: Mean / DC removal
       Python:
           a = X - X.mean(axis=1, keepdims=True)
       ========================================================= */

    float sum_raw = 0.0f;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        sum_raw += raw_samples[i];
    }

    float mean = sum_raw / (float)NUM_SAMPLES;


    /* =========================================================
       PASS 2: AC signal statistics
       ========================================================= */

    float energy_sum = 0.0f;
    float m3_sum = 0.0f;
    float m4_sum = 0.0f;

    float peak = 0.0f;

    float pmax = -FLT_MAX;
    float pmin =  FLT_MAX;

    float pos_energy = 0.0f;
    float neg_energy = 0.0f;

    for (int i = 0; i < NUM_SAMPLES; i++) {

        /* Remove DC */
        float a = raw_samples[i] - mean;

        float a2 = a * a;

        energy_sum += a2;
        m3_sum += a2 * a;
        m4_sum += a2 * a2;

        /* Absolute peak */
        float abs_a = fabsf(a);

        if (abs_a > peak) {
            peak = abs_a;
        }

        /* Positive / negative peak */
        if (a > pmax) {
            pmax = a;
        }

        if (a < pmin) {
            pmin = a;
        }

        /* Positive / negative energy */
        if (a > 0.0f) {
            pos_energy += a2;
        }
        else if (a < 0.0f) {
            neg_energy += a2;
        }
    }


    /* =========================================================
       RMS
       Python:
           e = (a ** 2).sum(axis=1)
           rms = sqrt(e / a.shape[1])
           safe_rms = where(rms > 1e-9, rms, 1.0)
       ========================================================= */

    float rms = sqrtf(
        energy_sum / (float)NUM_SAMPLES
    );

    float safe_rms =
        (rms > 1e-9f) ? rms : 1.0f;


    /* =========================================================
       Safe energy
       Python:
           safe_e = np.where(e > 1e-12, e, 1.0)
       ========================================================= */

    float safe_e =
        (energy_sum > 1e-12f)
        ? energy_sum
        : 1.0f;


    /* =========================================================
       FEATURE 0: Crest factor

       Python:
           peak = np.abs(a).max(axis=1)
           crest = peak / safe_rms
       ========================================================= */

    float crest =
        peak / safe_rms;


    /* =========================================================
       FEATURE 1: Skewness

       Python:
           skew = (a ** 3).mean(axis=1) / safe_rms ** 3
       ========================================================= */

    float third_moment =
        m3_sum / (float)NUM_SAMPLES;

    float skew =
        third_moment /
        (safe_rms * safe_rms * safe_rms);


    /* =========================================================
       FEATURE 2: Excess kurtosis

       Python:
           kurt = (a ** 4).mean(axis=1)
                  / safe_rms ** 4 - 3.0
       ========================================================= */

    float fourth_moment =
        m4_sum / (float)NUM_SAMPLES;

    float kurt =
        fourth_moment /
        (safe_rms * safe_rms * safe_rms * safe_rms)
        - 3.0f;


    /* =========================================================
       FEATURE 3: First-difference energy ratio

       Python:
           d1 = np.diff(a, axis=1)
           d1_ratio = (d1 ** 2).sum(axis=1) / safe_e

       Note:
       d1 can be calculated from raw_samples directly because
       the constant DC offset cancels:
           (x[i+1]-mean) - (x[i]-mean)
         = x[i+1]-x[i]
       ========================================================= */

    float d1_sum_sq = 0.0f;

    for (int i = 0; i < NUM_SAMPLES - 1; i++) {

        float diff1 =
            raw_samples[i + 1] -
            raw_samples[i];

        d1_sum_sq += diff1 * diff1;
    }

    float d1_ratio =
        d1_sum_sq / safe_e;


    /* =========================================================
       FEATURE 4: Second-difference energy ratio

       Python:
           d2 = a[:, 2:]
                - 2*a[:, 1:-1]
                + a[:, :-2]

           d2_ratio = (d2 ** 2).sum(axis=1) / safe_e
       ========================================================= */

    float d2_sum_sq = 0.0f;

    for (int i = 0; i < NUM_SAMPLES - 2; i++) {

        float diff2 =
            raw_samples[i + 2]
            - 2.0f * raw_samples[i + 1]
            + raw_samples[i];

        d2_sum_sq += diff2 * diff2;
    }

    float d2_ratio =
        d2_sum_sq / safe_e;


    /* =========================================================
       FEATURE 5: Peak asymmetry

       Python:
           pmax, pmin = a.max(axis=1), a.min(axis=1)
           span = where(
               (pmax - pmin) > 1e-12,
               pmax - pmin,
               1.0
           )
           peak_asym = (pmax + pmin) / span
       ========================================================= */

    float span = pmax - pmin;

    float safe_span =
        (span > 1e-12f)
        ? span
        : 1.0f;

    float peak_asym =
        (pmax + pmin) / safe_span;


    /* =========================================================
       FEATURE 6: Energy asymmetry

       Python:
           pos_e = where(a > 0, a**2, 0).sum(axis=1)
           neg_e = where(a < 0, a**2, 0).sum(axis=1)
           energy_asym = (pos_e - neg_e) / safe_e
       ========================================================= */

    float energy_asym =
        (pos_energy - neg_energy) / safe_e;


    /* =========================================================
       FEATURE 7: Fraction of small values

       IMPORTANT FIX

       Python:
           frac_small =
               (np.abs(a) < 0.1 * peak[:, None]).mean(axis=1)

       Therefore threshold = 10% OF PEAK.

       The previous C implementation incorrectly used:
           0.1 * std_dev

       which caused the large mismatch.
       ========================================================= */

    float small_threshold =
        0.1f * peak;

    int small_count = 0;

    for (int i = 0; i < NUM_SAMPLES; i++) {

        float a =
            raw_samples[i] - mean;

        if (fabsf(a) < small_threshold) {
            small_count++;
        }
    }

    float frac_small =
        (float)small_count /
        (float)NUM_SAMPLES;


    /* =========================================================
       FINAL FEATURE VECTOR
       Must exactly match Python order.
       ========================================================= */

    features_out[0] = crest;
    features_out[1] = skew;
    features_out[2] = kurt;
    features_out[3] = d1_ratio;
    features_out[4] = d2_ratio;
    features_out[5] = peak_asym;
    features_out[6] = energy_asym;
    features_out[7] = frac_small;
}