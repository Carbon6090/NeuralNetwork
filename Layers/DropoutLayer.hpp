#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include "Layer.hpp"

using namespace std;

class DropoutLayer : public Layer{
	default_random_engine generator;
	binomial_distribution<int> distribution;
	
	int total;
	double p;
	double q;
public:	
	DropoutLayer(TensorSize size, double p);

	void ForwardTrain(const Tensor &x);
	void Forward(const Tensor &x);
	void Backward(const Tensor &x, const Tensor &dout, bool needDx);
	void Save(ofstream &f);
	
	void Summary() const;
};

DropoutLayer::DropoutLayer(TensorSize size, double p) : Layer(size, size), distribution(1, 1 - p){
	total = size.height * size.width * size.depth;
	this->p = p;
	q = 1 - p;
}

void DropoutLayer::ForwardTrain(const Tensor &x) {
	for (int i = 0; i < total; i++){
		if (distribution(generator)){
			output[i] = x[i] / q;
			dx[i] = 1;
		}
		else{
			output[i] = 0;
			dx[i] = 0;
		}
	}
}

void DropoutLayer::Forward(const Tensor &x) {
	for (int i = 0; i < total; i++)
		output[i] = x[i];
}

void DropoutLayer::Backward(const Tensor &x, const Tensor &dout, bool needDx) {
	if (!needDx)
		return;

	for (int i = 0; i < total; i++)
		dx[i] *= dout[i];
}

void DropoutLayer::Save(ofstream &f){
	f << "dropout " << p << endl;
}

void DropoutLayer::Summary() const {
	cout << "|" << setw(22) << "dropout |" << setw(15) << inputSize << "|"<< setw(16) << outputSize << "|" << setw(14) << "0" << "| p: " << p << endl;
}