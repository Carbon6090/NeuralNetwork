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
	vector<outputs> output;
	vector<outputs> df;

	void InitializeWeights();
	double GetRnd(double a, double b);
public:	
	Layer(int inputs, int outputs, const string &function);
	vector<double> Forward(const vector<double> &x);
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
	
	InitializeWeights();
}

vector<double> Layer::Forward(const vector<double> &x){
	vector<double> z(outputs, 0);
	
	for (int i = 0; i < outputs; i++){
		double y = b[i];
		
		for (int j = 0; j < inputs; j++)
			y += w[i][j] * x[j];

			if (function == "sigmoid"){
				z[i] =  1.0 / (exp(-y) + 1);// sigmoid
				df[i] =  z[i] * (1 - z[i])
			}

			if (function == "tanh"){
				 z[i] = tanh(y); // tanh
				 df[i] = 1-z[i]*z[i];
			}

			if (function == "relu"){
				 z[i] = y > 0 ? y : 0; // relu z[i] = Activate(y);
				 df = z[i] > 0 ? 1 : 0;
			}
	}

	return z;
}