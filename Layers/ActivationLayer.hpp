#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include "Layer.hpp"

using namespace std;

class ActivationLayer : public Layer{
	string function; // "sigmoid" / "relu" / "tanh"
public:	
	ActivationLayer(int size, const string &function);

	void Forward(const vector<double> &x);
	void Backward(const vector<double> &x, const vector<double> &dout, bool needDx);

	void Print() const;
};

ActivationLayer::ActivationLayer(int size, const string &function) : Layer(size, size){
	this->function = function;
}

void ActivationLayer::Forward(const vector<double> &x) {	
	for (int i = 0; i < outputs; i++) {
		if (function == "sigmoid") {
			output[i] = 1.0 / (exp(-x[i]) + 1); // sigmoid
			dx[i] = output[i] * (1 - output[i]);
		}
		else if (function == "tanh") {
			output[i] = tanh(x[i]); // tanh
			dx[i] = 1 - output[i] * output[i];
		}
		else if (function == "relu"){
			output[i] = x[i] > 0 ? x[i] : 0; // relu
			dx[i] = output[i] > 0 ? 1 : 0;
		}
	}
}

void ActivationLayer::Backward(const vector<double> &x, const vector<double> &dout, bool needDx) {
	if (!needDx)
		return;
	
	for (int i = 0; i < outputs; i++)
		dx[i] *= dout[i];
}

void ActivationLayer::Print() const {
	cout << "Activation function: " << function << endl;
}