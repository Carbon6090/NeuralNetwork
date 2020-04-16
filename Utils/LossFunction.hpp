#pragma once
#include <iostream>

typedef double (*LossFunction)(const Tensor &y, const Tensor &t, Tensor &dout);

double MSE(const Tensor &y, const Tensor &t, Tensor &dout){
	double loss = 0;

	for (int i = 0; i < y.Total(); i++){
		double e = y[i] - t[i];
		loss += e * e;
		dout[i] = 2 * e;
	}

	return loss;
}

double CrossEntropy(const Tensor &y, const Tensor &t, Tensor &dout){
	double loss = 0;

	for (int i = 0; i < y.Total(); i++){
		dout[i] = -t[i] / y[i];
		loss -= t[i] * log(y[i]);
	}

	return loss;
}

double BinaryCrossEntropy(const Tensor &y, const Tensor &t, Tensor &dout){
	double loss = 0;

	for (int i = 0; i < y.Total(); i++){
		dout[i] = -(y[i] - t[i]) / (y[i] - y[i] * y[i]);
		loss -= t[i]*log(y[i]) + (1 - t[i])*log(1-y[i]);
	}

	return loss;
}