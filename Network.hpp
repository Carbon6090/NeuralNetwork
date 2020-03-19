#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include "Layer.hpp"

using namespace std;

class Network{
	int inputs;
	int last;
	vector<Layer> layers;
	void Backward(const vector<double> &x, const vector<double> &dout);
	double CalculateLoss(const vector<double> &y, const vector<double> &t, vector<double> &dout);
	void UpdateWeights(double learningRate);
public:
	Network(int inputs);
	void AddLayer(int outputs, const string& function);
	vector<double> Forward(const vector<double> &x);
	void Train(const vector<vector<double>> &x, const vector<vector<double>> &y, double learningRate, int epochs, int period);
	void Print() const;
};

void Network::Backward(const vector<double> &x, const vector<double> &dout){
	if (last == 0){
		layers[last].Backward(x, dout);
		return;
	}

	layers[last].Backward(layers[last - 1].GetOutput(), dout);
	for (int i = last - 1; i >= 1; i--)
		layers[i].Backward(layers[i - 1].GetOutput(), layers[i + 1].GetDx());

	layers[0].Backward(x, layers[1].GetDx());
}

double Network::CalculateLoss(const vector<double> &y, const vector<double> &t, vector<double> &dout){
	double loss = 0;

	for (int i = 0; i < y.size(); i++){
		double e = y[i] - t[i];
		loss += e * e;
		dout[i] = 2 * e;
	}

	return loss;
}

void Network::UpdateWeights(double learningRate){
	for (int i = 0; i <= last; i++)
		layers[i].UpdateWeights(learningRate);
}

Network::Network(int inputs){
	this->inputs = inputs;
	last = -1;
}

void Network::AddLayer(int outputs, const string& function){
	int inputsSize = layers.size() ? layers[last].GetSize() : inputs;
	layers.push_back(Layer(inputsSize, outputs, function));
	last++;
}

vector<double> Network::Forward(const vector<double> &x){
	layers[0].Forward(x);

	for (int i = 1; i < layers.size(); i++)
		layers[i].Forward(layers[i - 1].GetOutput());

	return layers[last].GetOutput();
}

void Network::Train(const vector<vector<double>> &x, const vector<vector<double>> &y, double learningRate, int epochs, int period){
	for (int epoch = 0; epoch < epochs; epoch++){
		double loss = 0;
		
		for (int i = 0; i < x.size(); i++){
			vector<double> out = Forward(x[i]);
			vector<double> dout(y[i].size());

			loss += CalculateLoss(out, y[i], dout);
			Backward(x[i], dout);
			UpdateWeights(learningRate);
		}
		
		if (epoch % period == 0)
			cout << "Epoch: " << epoch << ", loss: " << loss << endl;
	}
}

void Network::Print() const {
	for (int i = 0; i < layers.size(); i++){
		cout << "layer " << i << ": " << endl;
		layers[i].Print();
	}
}