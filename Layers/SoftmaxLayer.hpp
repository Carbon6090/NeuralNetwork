#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "Layer.hpp"

using namespace std;

class SoftmaxLayer : public Layer{
public:	
	SoftmaxLayer(int size);

	void Forward(const vector<double> &x);
	void Backward(const vector<double> &x, const vector<double> &dout, bool needDx);
	
	void Summary() const;
};

SoftmaxLayer::SoftmaxLayer(int size) : Layer(size, size){}

void SoftmaxLayer::Forward(const vector<double> &x) {	
	double sum = 0;

	for (int i = 0; i < outputs; i++){
		output[i] = exp(x[i]);
		sum += output[i];
	}

	for (int i = 0; i < outputs; i++) 
		output[i] /= sum;
}

void SoftmaxLayer::Backward(const vector<double> &x, const vector<double> &dout, bool needDx) {
	for (int i = 0; i < outputs; i++) {
		dx[i] = 0;

		for (int j = 0; j < outputs; j++)
			dx[i] += dout[j] * output[i] * ((i == j) - output[j]);
	}
}

void SoftmaxLayer::Summary() const {
	cout << "|" << setw(22) << "softmax layer|" << setw(15) << inputs << "|"<< setw(16) << outputs << "|" << setw(14) << "0" << "|" << endl;
}