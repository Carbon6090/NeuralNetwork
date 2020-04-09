#pragma once
#include <iostream>

typedef double (*LossFunction)(const vector<double> &y, const vector<double> &t, vector<double> &dout);

double MSE(const vector<double> &y, const vector<double> &t, vector<double> &dout){
	double loss = 0;

	for (int i = 0; i < y.size(); i++){
		double e = y[i] - t[i];
		loss += e * e;
		dout[i] = 2 * e;
	}

	return loss;
}

double CrossEntropy(const vector<double> &y, const vector<double> &t, vector<double> &dout){
	double loss = 0;

	for (int i = 0; i < y.size(); i++){
		dout[i] = -t[i] / y[i];
		loss -= t[i] * log(y[i]);
	}

	return loss;
}

double BinaryCrossEntropy(const vector<double> &y, const vector<double> &t, vector<double> &dout){
	double loss = 0;

	for (int i = 0; i < y.size(); i++){
		dout[i] = -(y[i] - t[i]) / (y[i] - y[i] * y[i]);
		loss -= t[i]*log(y[i]) + (1 - t[i])*log(1-y[i]);
	}

	return loss;
}