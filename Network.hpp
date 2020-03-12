#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include "Layer.hpp"

using namespace std;

class Network{
	int inputs;
	vector<Layer> layers;
public:
	Network(int inputs);
	void AddLayer(int outputs, const string& function);
	vector<double> Forward(const vector<double> &x);
};

Network::Network(int inputs){
	this->inputs = inputs;
}

void Network::AddLayer(int outputs, const string& function){
	int inputsSize = layers.size() ? layers[layers.size() - 1].GetSize() : inputs;
	layers.push_back(Layer(inputsSize, outputs, function));
}

vector<double> Network::Forward(const vector<double> &x){
	layers[0].Forward(x);

	for (int i = 1; i < layers.size(); i++)
		layers[i].Forward(layers[i - 1].GetOutput());

	return layers[layers.size() - 1].GetOutput();
}