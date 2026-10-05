#include <stdio.h>
#include "tinyml_engine.h"

int main() {
    // Normal / Baseline load sample vector (Low noise, normal current metrics)
    float normal_features[14] = {
        0.15f, 0.05f, 0.15f, 0.30f, 0.45f, 1.41f,
        0.02f, 3.01f, 12000.0f, 11500.0f,
        300.0f, 200.0f, 0.001f, 450.0f
    };

    float probability = 0.0f;
    int prediction = tinyml_predict(normal_features, &probability);

    printf("====================================================\n");
    printf("TINYML C ENGINE NORMAL SAMPLE TEST\n");
    printf("====================================================\n");
    printf("Predicted Class       : %s\n", prediction == 1 ? "ARC FAULT (1)" : "NORMAL (0)");
    printf("Arc Fault Probability : %.4f (%.2f%%)\n", probability, probability * 100.0f);
    printf("====================================================\n");

    return 0;
}