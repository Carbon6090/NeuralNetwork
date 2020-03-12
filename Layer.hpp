#pragma once
#include <iostream>
#include <vector>

using namespace std;

class Layer{
	int inputs;
	int outputs;
	string function;
	vector<vector<double>> w;
	vector<double> b;
	vector<double> output;
	vector<double> df;

	void InitializeWeights();
	double GetRnd(double a, double b);
public:	
	Layer(int inputs, int outputs, const string &function);
	void Forward(const vector<double> &x);
};

void Layer::InitializeWeights(){
	w = vector<vector<double>>(outputs, vector<double>(inputs));
	b = vector<double>(outputs);
	
	for (int i = 0; i < outputs; i++){
		for (int j = 0; j < inputs; j++)
			w[i][j] = GetRnd(-0.5, 0.5);

		b[i] = GetRnd(-0.5, 0.5); 
	}
}

double Layer::GetRnd(double a, double b) {
	return a + ((b - a) * rand()) / RAND_MAX;
}

Layer::Layer(int inputs, int , const string &function) {
	this->inputs = inputs;
	this->outputs = outputs;
	this->function = function;
	output = vector<double>(outputs, 0);
	df = vector<double>(outputs, 0);

	InitializeWeights();
}

void Layer::Forward(const vector<double> &x){
	
	for (int i = 0; i < outputs; i++){
		double y = b[i];
		
		for (int j = 0; j < inputs; j++)
			y += w[i][j] * x[j];

			if (function == "sigmoid"){
				output[i] =  1.0 / (exp(-y) + 1);// sigmoid
				df[i] =  output[i] * (1 - z[i])
			}

			else if (function == "tanh"){
				 output[i] = tanh(y); // tanh
				 df[i] = 1 - output[i] * output[i];
			}

			else if (function == "relu"){
				 output[i] = y > 0 ? y : 0; // relu z[i] = Activate(y);
				 df = output[i] > 0 ? 1 : 0;
			}
	}

	return output;
}