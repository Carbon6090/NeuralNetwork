#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include "Layer.hpp"

using namespace std;

class FullConnectedLayer : public Layer{
	vector<vector<double>> w;
	vector<double> b;

	vector<vector<double>> dw;
	vector<double> db;

	void InitializeWeights();
public:	
	FullConnectedLayer(int inputs, int output);

	void Forward(const vector<double> &x);
	void Backward(const vector<double> &x, const vector<double> &dout, bool needDx);
	void UpdateWeights(double learningRate);
	
	void Print() const;
};

void FullConnectedLayer::InitializeWeights(){	
	for (int i = 0; i < outputs; i++){
		for (int j = 0; j < inputs; j++)
			w[i][j] = GetRnd(-0.5, 0.5);

		b[i] = 0;//GetRnd(-0.5, 0.5); 
	}
}

FullConnectedLayer::FullConnectedLayer(int inputs, int outputs) : Layer(inputs, outputs){
	w = vector<vector<double>>(outputs, vector<double>(inputs));
	b = vector<double>(outputs);

	dw = vector<vector<double>>(outputs, vector<double>(inputs));
	db = vector<double>(outputs);

	InitializeWeights();
}

void FullConnectedLayer::Forward(const vector<double> &x) {	
	for (int i = 0; i < outputs; i++) {
		double y = b[i];
		
		for (int j = 0; j < inputs; j++)
			y += w[i][j] * x[j];

		output[i] = y;
	}
}

void FullConnectedLayer::Backward(const vector<double> &x, const vector<double> &dout, bool needDx) {
	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			dw[i][j] = dout[i] * x[j];

		db[i] = dout[i];
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
	for (int i = 0; i < outputs; i++){
		for (int j = 0; j < inputs; j++)
			w[i][j] -= learningRate * dw[i][j];

		b[i] -= learningRate * db[i];
	}
}

void FullConnectedLayer::Print() const {
	cout << "Full connected layer" << endl;
}