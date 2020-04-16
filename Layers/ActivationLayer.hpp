#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "Layer.hpp"

using namespace std;

class ActivationLayer : public Layer{
	string function; // "sigmoid" / "relu" / "tanh"
	int total;
public:	
	ActivationLayer(TensorSize size, const string &function);
	ActivationLayer(TensorSize size, ifstream &f);

	void Forward(const Tensor &x);
	void Backward(const Tensor &x, const Tensor &dout, bool needDx);
	void Save(ofstream &f);

	void Summary() const;
};

ActivationLayer::ActivationLayer(TensorSize size, const string &function) : Layer(size, size){
	this->function = function;
	total = size.height * size.width * size.depth;
}

ActivationLayer::ActivationLayer(TensorSize size, ifstream &f) : Layer(size, size){
	f >> function;
	total = size.height * size.width * size.depth;
}


void ActivationLayer::Forward(const Tensor &x) {	
	for (int i = 0; i < total; i++) {
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

void ActivationLayer::Backward(const Tensor &x, const Tensor &dout, bool needDx) {
	if (!needDx)
		return;
	
	for (int i = 0; i < total; i++)
		dx[i] *= dout[i];
}

void ActivationLayer::Save(ofstream &f){
	f << "activation " << " " << function << endl;
}

void ActivationLayer::Summary() const {
	string name = "activation '" + function + "'";
	cout << "|" << setw(21) << name << "|" << setw(15) << inputSize << "|" << setw(16) << outputSize << "|" << setw(14) << "0" << "|" << endl;
}