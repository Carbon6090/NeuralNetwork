#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "Layer.hpp"

using namespace std;

class SoftmaxLayer : public Layer{
	int total;
public:	
	SoftmaxLayer(TensorSize size);

	void Forward(const Tensor &x);
	void Backward(const Tensor &x, const Tensor &dout, bool needDx);
	void Save(ofstream &f);
	
	void Summary() const;
};

SoftmaxLayer::SoftmaxLayer(TensorSize size) : Layer(size, size){
	total = size.height * size.width * size.depth;
}

void SoftmaxLayer::Forward(const Tensor &x) {	
	double sum = 0;

	for (int i = 0; i < total; i++){
		output[i] = exp(x[i]);
		sum += output[i];
	}

	for (int i = 0; i < total; i++) 
		output[i] /= sum;
}

void SoftmaxLayer::Backward(const Tensor &x, const Tensor &dout, bool needDx) {
	for (int i = 0; i < total; i++) {
		dx[i] = 0;

		for (int j = 0; j < total; j++)
			dx[i] += dout[j] * output[i] * ((i == j) - output[j]);
	}
}

void SoftmaxLayer::Save(ofstream &f){
	f << "softmax " << endl;
}

void SoftmaxLayer::Summary() const {
	cout << "|" << setw(22) << "softmax |" << setw(15) << inputSize << "|"<< setw(16) << outputSize << "|" << setw(14) << "0" << "|" << endl;
}