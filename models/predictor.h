#pragma once

const int N_FEATURES = 15;
extern const double mean[N_FEATURES];
extern const double scale[N_FEATURES];
void scale_input(double* x);
double predict(const double* x);
