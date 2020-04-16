#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include "Layer.hpp"

using namespace std;

class FullConnectedLayer : public Layer{
	default_random_engine generator;
	normal_distribution<double> distribution;

	int inputs;
	int outputs;

	vector<vector<double>> w;
	vector<vector<double>> dw;

	void InitializeWeights();
public:	
	FullConnectedLayer(TensorSize inputs, int size);
	FullConnectedLayer(TensorSize inputs, int size, ifstream &f);

	void Forward(const Tensor &x);
	void Backward(const Tensor &x, const Tensor &dout, bool needDx);
	void UpdateWeights(double learningRate);
	void Save(ofstream &f);

	void Summary() const;
};

void FullConnectedLayer::InitializeWeights() {	
	for (int i = 0; i < outputs; i++)
		for (int j = 0; j <= inputs; j++)
			w[i][j] = distribution(generator);
}

FullConnectedLayer::FullConnectedLayer(TensorSize inputSize, int size) : Layer(inputSize, { 1, 1, size }), distribution(0.0, sqrt(2.0 / (inputSize.height * inputSize.width * inputSize.depth))) {
	inputs = inputSize.height * inputSize.width * inputSize.depth;
	outputs = size;

	w = vector<vector<double>>(outputs, vector<double>(inputs + 1));
	dw = vector<vector<double>>(outputs, vector<double>(inputs + 1));

	InitializeWeights();
}

FullConnectedLayer::FullConnectedLayer(TensorSize inputSize, int size, ifstream &f) : Layer(inputSize, { 1, 1, size }), distribution(0.0, sqrt(2.0 / (inputSize.height * inputSize.width * inputSize.depth))) {
	inputs = inputSize.height * inputSize.width * inputSize.depth;
	outputs = size;

	w = vector<vector<double>>(outputs, vector<double>(inputs + 1));
	dw = vector<vector<double>>(outputs, vector<double>(inputs + 1));

	for (int i = 0; i < outputs; i++)
		for (int j = 0; j <= inputs; j++)
			f >> w[i][j];
}


void FullConnectedLayer::Forward(const Tensor &x) {	
	for (int i = 0; i < outputs; i++) {
		double y = w[i][inputs];
		
		for (int j = 0; j < inputs; j++)
			y += w[i][j] * x[j];

		output[i] = y;
	}
}

void FullConnectedLayer::Backward(const Tensor &x, const Tensor &dout, bool needDx) {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			dw[i][j] = dout[i] * x[j];

		dw[i][inputs] = dout[i];
	}

	if (!needDx)
		return;

	for (int i = 0; i < inputs; i++) {
		double dx_i = 0;
		for (int j = 0; j < outputs; j++)
			dx_i += w[j][i] * dout[j];

		dx[i] = dx_i;
	}
}

void FullConnectedLayer::UpdateWeights(double learningRate) {
	for (int i = 0; i < outputs; i++)
		for (int j = 0; j <= inputs; j++)
			w[i][j] -= learningRate * dw[i][j];
}

void FullConnectedLayer::Save(ofstream &f){
	f << "fc " << outputs << endl;

	for (int i = 0; i < outputs; i++){
		for (int j = 0; j <= inputs; j++)
			f << w[i][j] << " ";

		f << endl;
	}
}

void FullConnectedLayer::Summary() const {
	cout << "|" << setw(22) << "fc layer|" << setw(15) << inputSize << "|" << setw(16) << outputSize << "|" << setw(14) << (inputs + 1) * outputs<< "|" << endl;
}