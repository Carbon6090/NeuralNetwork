#pragma once
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class Layer{
	int inputs;
	int outputs;
	string function; // "sigmoid" / "relu" / "tanh"
	vector<vector<double>> w;
	vector<double> b;
	vector<double> output;

	vector<vector<double>> dw;
	vector<double> db;
	vector<double> df;
	vector<double> dx;

	void InitializeWeights();
	double GetRnd(double a, double b);
public:	
	Layer(int inputs, int outputs, const string &function);

	vector<double> Forward(const vector<double> &x);
	vector<double> Backward(const vector<double> &x, const vector<double> &dout);
	void UpdateWeights(double learningRate);
	
	vector<double> GetOutput() const;
	int GetSize() const;
};

void Layer::InitializeWeights(){	
	for (int i = 0; i < outputs; i++){
		for (int j = 0; j < inputs; j++)
			w[i][j] = GetRnd(-0.5, 0.5);

		b[i] = GetRnd(-0.5, 0.5); 
	}
}

double Layer::GetRnd(double a, double b) {
	return a + ((b - a) * rand()) / RAND_MAX;
}

Layer::Layer(int inputs, int outputs, const string &function) {
	this->inputs = inputs;
	this->outputs = outputs;
	this->function = function;

	w = vector<vector<double>>(outputs, vector<double>(inputs));
	b = vector<double>(outputs);
	output = vector<double>(outputs, 0);

	dw = vector<vector<double>>(outputs, vector<double>(inputs));
	db = vector<double>(outputs);
	df = vector<double>(outputs, 0);
	dx = vector<double>(inputs, 0);

	InitializeWeights();
}

vector<double> Layer::Forward(const vector<double> &x) {	
	for (int i = 0; i < outputs; i++) {
		double y = b[i];
		
		for (int j = 0; j < inputs; j++)
			y += w[i][j] * x[j];

		if (function == "sigmoid") {
			output[i] = 1.0 / (exp(-y) + 1); // sigmoid
			df[i] = output[i] * (1 - output[i]);
		}
		else if (function == "tanh") {
			output[i] = tanh(y); // tanh
			df[i] = 1 - output[i] * output[i];
		}
		else if (function == "relu"){
			output[i] = y > 0 ? y : 0; // relu
			df[i] = output[i] > 0 ? 1 : 0;
		}
	}

	return output;
}

vector<double> Layer::Backward(const vector<double> &x, const vector<double> &dout) {
	for (int i = 0; i < outputs; i++)
		df[i] *= dout[i];

	for (int i = 0; i < outputs; i++) {
		for (int j = 0; j < inputs; j++)
			dw[i][j] = df[i] * x[j];

		db[i] = df[i];
	}

	for (int i = 0; i < inputs; i++) {
		double dx_i = 0;
		for (int j = 0; j < outputs; j++)
			dx_i += w[j][i] * df[j];

		dx[i] = dx_i;
	}

	return dx;
}

void Layer::UpdateWeights(double learningRate) {
	for (int i = 0; i < outputs; i++){
		for (int j = 0; j < inputs; j++)
			w[i][j] -= learningRate * dw[i][j];

		b[i] -= learningRate * db[i];
	}
}

vector<double> Layer::GetOutput() const {
	return output;
}

int Layer::GetSize() const {
	return outputs;
}