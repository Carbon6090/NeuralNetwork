#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

class Layer{
protected:
	TensorSize outputSize;
	TensorSize inputSize;
	Tensor output;
	Tensor dx;
public:	
	Layer(TensorSize inputSize, TensorSize outputSize);

	virtual void Forward(const Tensor &x) = 0;
	virtual void ForwardTrain(const Tensor &x);
	virtual void Backward(const Tensor &x, const Tensor &dout, bool needDx) = 0;
	virtual void UpdateWeights(double learningRate);
	virtual void Save(ofstream &f) = 0;
	virtual void Load(ifstream &f);
	
	TensorSize GetOutputSize() const;
	Tensor GetOutput() const;
	Tensor GetDx() const;
	virtual void Summary() const = 0;
};

Layer::Layer(TensorSize inputSize, TensorSize outputSize) : output(outputSize), dx(inputSize){
	this->inputSize = inputSize;
	this->outputSize = outputSize;
}

void Layer::ForwardTrain(const Tensor &x){
	Forward(x);
}

void Layer::UpdateWeights(double learningRate) {
}

void Layer::Load(ifstream &f) {
}

TensorSize Layer::GetOutputSize() const{
	return outputSize;
}

Tensor Layer::GetOutput() const {
	return output;
}

Tensor Layer::GetDx() const {
	return dx;
}