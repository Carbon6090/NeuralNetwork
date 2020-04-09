#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include "Layers/Layer.hpp"
#include "Layers/FullConnectedLayer.hpp"
#include "Layers/ActivationLayer.hpp"
#include "Layers/SoftmaxLayer.hpp"
#include "LossFunction.hpp"

using namespace std;

struct Data{
	vector<vector<double>> x;
	vector<vector<double>> y;
};

class Network{
	int inputs;
	int outputs;
	int last;
	vector<Layer*> layers;
	void Backward(const vector<double> &x, const vector<double> &dout);
	void UpdateWeights(double learningRate);
public:
	Network(int inputs);
	void AddLayer(const string& description);
	vector<double> Forward(const vector<double> &x);
	void Train(const Data &data, double learningRate, int epochs, int period, LossFunction L);
	void Summary() const;
};

void Network::Backward(const vector<double> &x, const vector<double> &dout){
	if (last == 0){
		layers[last]->Backward(x, dout, false);
		return;
	}

	layers[last]->Backward(layers[last - 1]->GetOutput(), dout, true);
	for (int i = last - 1; i >= 1; i--)
		layers[i]->Backward(layers[i - 1]->GetOutput(), layers[i + 1]->GetDx(), true);

	layers[0]->Backward(x, layers[1]->GetDx(), false);
}

void Network::UpdateWeights(double learningRate){
	for (int i = 0; i <= last; i++)
		layers[i]->UpdateWeights(learningRate);
}

Network::Network(int inputs){
	this->inputs = inputs;
	outputs = -1;
	last = -1;
}

// fc size / activation function 
void Network::AddLayer(const string& description){
	int inputsSize = layers.size() ? this->outputs : inputs;
	
	stringstream ss(description);
	string name;
	ss >> name;
	
	if (name == "fc" || name == "fullconnected") {
		int size;
		ss >> size;
		layers.push_back(new FullConnectedLayer(inputsSize, size));
		this->outputs = size;
	}
	else if (name == "activation"){
		string function;
		ss >> function;
		layers.push_back(new ActivationLayer(inputsSize, function));
	}
	else if (name == "softmax"){
		layers.push_back(new SoftmaxLayer(inputsSize));
	}
	else
		throw runtime_error("Unknown layer: " + name);

	last++;
}

vector<double> Network::Forward(const vector<double> &x){
	layers[0]->Forward(x);

	for (int i = 1; i < layers.size(); i++)
		layers[i]->Forward(layers[i - 1]->GetOutput());

	return layers[last]->GetOutput();
}

void Network::Train(const Data &data, double learningRate, int epochs, int period, LossFunction L){
	for (int epoch = 0; epoch < epochs; epoch++){
		double loss = 0;
		
		for (int i = 0; i < data.x.size(); i++){
			vector<double> out = Forward(data.x[i]);
			vector<double> dout(data.y[i].size());

			loss += L(out, data.y[i], dout);
			Backward(data.x[i], dout);
			UpdateWeights(learningRate);
		}
		
		if (epoch % period == 0)
			cout << "Epoch: " << epoch << ", loss: " << loss << endl;
	}
}

void Network::Summary() const {
	cout << "+---------------------+---------------+----------------+--------------+" << endl;
	cout << "|     layer name      |  inputs count |  outputs count | weghts count |" << endl;
	cout << "+---------------------+---------------+----------------+--------------+" << endl;

	for (int i = 0; i < layers.size(); i++)
		layers[i]->Summary();

	cout << "+---------------------+---------------+----------------+--------------+" << endl;
}