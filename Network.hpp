#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <chrono>
#include "Layers/Layer.hpp"
#include "Layers/FullConnectedLayer.hpp"
#include "Layers/ActivationLayer.hpp"
#include "Layers/SoftmaxLayer.hpp"
#include "Layers/DropoutLayer.hpp"
#include "Layers/MaxPoolingLayer.hpp"
#include "Layers/ConvolutionLayer.hpp"
#include "Utils/Data.hpp"
#include "Utils/LossFunction.hpp"
#include "Utils/Tensor.hpp"

using namespace std;
using namespace std::chrono;

typedef high_resolution_clock Time;
typedef time_point<Time> TimePoint; 
typedef milliseconds ms;

class Network{
	TensorSize inputSize;
	TensorSize outputSize;
	int last;
	vector<Layer*> layers;
	Tensor ForwardTrain(const Tensor &x);
	void Backward(const Tensor &x, const Tensor &dout);
	void UpdateWeights(double learningRate);
public:
	Network(TensorSize inputSize);
	Network(const string &path);
	void AddLayer(const string& description);
	Tensor Forward(const Tensor &x);
	void Train(const Data &data, double learningRate, int epochs, int period, LossFunction L);
	void Summary() const;
	void Save(const string &path);
};

Tensor Network::ForwardTrain(const Tensor &x) {
	layers[0]->ForwardTrain(x);

	for (int i = 1; i < layers.size(); i++)
		layers[i]->ForwardTrain(layers[i - 1]->GetOutput());

	return layers[last]->GetOutput();
}

void Network::Backward(const Tensor &x, const Tensor &dout) {
	if (last == 0) {
		layers[last]->Backward(x, dout, false);
		return;
	}

	layers[last]->Backward(layers[last - 1]->GetOutput(), dout, true);
	for (int i = last - 1; i >= 1; i--)
		layers[i]->Backward(layers[i - 1]->GetOutput(), layers[i + 1]->GetDx(), true);

	layers[0]->Backward(x, layers[1]->GetDx(), false);
}

void Network::UpdateWeights(double learningRate) {
	for (int i = 0; i <= last; i++)
		layers[i]->UpdateWeights(learningRate);
}

Network::Network(TensorSize inputSize) {
	this->inputSize = inputSize;
	outputSize = inputSize;
	last = -1;
}

Network::Network(const string &path) {
	ifstream f(path);

	if (!f)
		throw runtime_error("invalid file");
	
	f >> inputSize.height >> inputSize.width >> inputSize.depth;
	outputSize = inputSize;

	last = -1;
	
	while (!f.eof()) {
		string name;	
		f >> name;

		if(name == "")
			continue;

		TensorSize inputs = outputSize;

		if (name == "activation") {
			layers.push_back(new ActivationLayer(inputs, f));
		}
		else if (name == "fc" || name == "fullconnected") {
			int outputs;
			f >> outputs;

			layers.push_back(new FullConnectedLayer(inputs, outputs));
		}
		else if (name == "softmax") {
			layers.push_back(new SoftmaxLayer(inputs));
		}
		else if (name == "dropout") {
			double p;
			f >> p;
			layers.push_back(new DropoutLayer(inputs, p));
		}
		else if (name == "maxpooling") {
			double scale;
			f >> scale;
			layers.push_back(new MaxPoolingLayer(inputs, scale));
		}
		else if (name == "convolution") {
			int fc, fs, padding;
			f >> fc >> fs >> padding;
			
			layers.push_back(new ConvolutionLayer(inputs, fc, fs, padding));
		}
		else
			throw runtime_error("Unknown layer '" + name + "'");

		outputSize = layers[++last]->GetOutputSize();

		layers[last]->Load(f);
	}
}

// fc size / activation function 
void Network::AddLayer(const string& description) {
	TensorSize inputs = outputSize;
	
	stringstream ss(description);
	string name;
	ss >> name;
	
	if (name == "fc" || name == "fullconnected") {
		int size;
		ss >> size;

		layers.push_back(new FullConnectedLayer(inputs, size));
	}
	else if (name == "activation") {
		string function;
		ss >> function;

		layers.push_back(new ActivationLayer(inputs, function));
	}
	else if (name == "softmax") {
		layers.push_back(new SoftmaxLayer(inputs));
	}
	else if (name == "dropout") {
		double p;
		ss >> p;

		layers.push_back(new DropoutLayer(inputs, p));
	}
	else if (name == "maxpooling") {
		double scale;
		ss >> scale;

		layers.push_back(new MaxPoolingLayer(inputs, scale));
	}
	else if (name == "convolution") {
		int fc, fs, padding;
		ss >> fc >> fs >> padding;

		layers.push_back(new ConvolutionLayer(inputs, fc, fs, padding));
	}
	else
		throw runtime_error("Unknown layer: " + name);

	outputSize = layers[++last]->GetOutputSize();
}

Tensor Network::Forward(const Tensor &x) {
	layers[0]->Forward(x);

	for (int i = 1; i < layers.size(); i++)
		layers[i]->Forward(layers[i - 1]->GetOutput());

	return layers[last]->GetOutput();
}

void Network::Train(const Data &data, double learningRate, int epochs, int period, LossFunction L) {
	for (int epoch = 0; epoch < epochs; epoch++) {
		double loss = 0;
		TimePoint t0 = Time::now();
		
		for (int i = 0; i < data.x.size(); i++) {
			Tensor out = ForwardTrain(data.x[i]);
			Tensor dout(outputSize);

			loss += L(out, data.y[i], dout);
			Backward(data.x[i], dout);
			UpdateWeights(learningRate);
		}
		
		loss /= data.x.size();
		TimePoint t1 = Time::now();

		if (epoch % period == 0) {
			ms dt = duration_cast<ms>(t1 - t0);
			cout << "Epoch: " << epoch << ", loss: " << loss << ", time: " << dt.count() / 1000.0 << " seconds" << endl;
		}
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

void Network::Save(const string &path) {
	ofstream f(path);
	f << inputSize.height << " " << inputSize.width << " " << inputSize.depth << endl;

	for (int i = 0; i < layers.size(); i++)
		layers[i]->Save(f);

	f.close();
}