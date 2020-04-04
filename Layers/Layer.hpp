#pragma once
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class Layer{
protected:
	int inputs;
	int outputs;

	vector<double> output;
	vector<double> dx;

	double GetRnd(double a, double b);
public:	
	Layer(int inputs, int outputs);

	virtual void Forward(const vector<double> &x) = 0;
	virtual void Backward(const vector<double> &x, const vector<double> &dout) = 0;
	virtual void UpdateWeights(double learningRate);
	
	vector<double> GetOutput() const;
	vector<double> GetDx() const;
	virtual void Print() const = 0;
};

double Layer::GetRnd(double a, double b) {
	return a + ((b - a) * rand()) / RAND_MAX;
}

Layer::Layer(int inputs, int outputs) {
	this->inputs = inputs;
	this->outputs = outputs;

	output = vector<double>(outputs, 0);
	dx = vector<double>(inputs, 0);
}

void Layer::UpdateWeights(double learningRate) {
}

vector<double> Layer::GetOutput() const {
	return output;
}

vector<double> Layer::GetDx() const {
	return dx;
}