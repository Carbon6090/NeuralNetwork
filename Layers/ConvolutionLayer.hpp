#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "Layer.hpp"

class ConvolutionLayer : public Layer {
	default_random_engine generator;
	normal_distribution<double> distribution;

	TensorSize outputSize;

	vector<Tensor> w;
	vector<Tensor> dw;

	vector<double> b;
	vector<double> db;

	int fc;
	int fs;
	int fd;
	int padding;

	void InitWeights();
public:
	ConvolutionLayer(TensorSize inputSize, int fc, int fs, int padding);

	void Forward(const Tensor &x);
	void Backward(const Tensor &x, const Tensor &dout, bool needDx);

	void UpdateWeights(double learningRate);
	
	void Save(ofstream &f);
	void Load(ifstream &f);
	void Summary() const;
};

void ConvolutionLayer::InitWeights() {	
	for (int i = 0; i < fc; i++) {
		b[i] = distribution(generator);
		db[i] = distribution(generator);
		
		for (int j = 0; j < w[i].Total(); j++) {
			w[i][j] = distribution(generator);
			dw[i][j] = distribution(generator);
		}
	}
}

ConvolutionLayer::ConvolutionLayer(TensorSize size, int fc, int fs, int padding) : Layer (size, {(size.height - fs + 2 * padding + 1), (size.width - fs + 2 * padding + 1), fc}), distribution(0.0, sqrt(2.0 / (fs * fs * size.depth))) {
	this->fc = fc;
	this->fs = fs;
	fd = size.depth;
	this->padding = padding;

	for (int i = 0; i < fc; i++) {
		w.push_back(Tensor(fs, fs, fd));
		dw.push_back(Tensor(fs, fs, fd));
	}

	b = vector<double>(fc, 0);
	db = vector<double>(fc, 0);

	InitWeights();

	outputSize.height = size.height - fs + 2 * padding + 1;
	outputSize.width = size.width - fs + 2 * padding + 1;
	outputSize.depth = fc;
}

void ConvolutionLayer::Forward(const Tensor &x) {
	for (int f = 0; f < fc; f++) {
		for (int i = 0; i < outputSize.height; i++) {
			for (int j = 0; j < outputSize.width; j++) {
				double sum = b[f];

				for (int k = 0; k < fs; k++) {
					int i0 = i + k - padding;
					
					if (i0 < 0 || i0 >= inputSize.height)
						continue;

					for (int l = 0; l < fs; l++) {
						int j0 = j + l - padding;

						if (j0 < 0 || j0 >= inputSize.width)
							continue;

						for (int d = 0; d < fd; d++)
							sum += x(i0, j0, d) * w[f](k, l, d);
					}
				}

				output(i, j, f) = sum;
			}
		}		
	}
}

void ConvolutionLayer::Backward(const Tensor &x, const Tensor &dout, bool needDx) {
	for (int f = 0; f < fc; f++) {
		for (int k = 0; k < outputSize.height; k++) {
			for (int l = 0; l < outputSize.width; l++) {
				double delta = dout(k, l, f);

				for (int i = 0; i < fs; i++) {
					int i0 = i + k - padding;

					if (i0 < 0 || i0 >= inputSize.height)
					continue;

					for (int j = 0; j < fs; j++) {
						int j0 = j + l - padding;

						if (j0 < 0 || j0 >= inputSize.width)
							continue;

						for (int c = 0; c < fd; c++)
							dw[f](i, j, c) += delta * x(i0, j0, c);
					}
				}

				db[f] += delta;
			}
		}
	}

	if (!needDx)
		return;

	int pad = fs - 1 - padding;

	for (int i = 0; i < inputSize.height; i++) {
		for (int j = 0; j < inputSize.width; j++) {
			for (int c = 0; c < fd; c++) {
				double sum = 0;

				for (int k = 0; k < fs; k++) {
					int i0 = i + k - pad;

					if (i0 < 0 || i0 >= outputSize.height)
						continue;

					for (int l = 0; l < fs; l++) {
						int j0 = j + l - pad;

						if (j0 < 0 || j0 >= outputSize.width)
							continue;

						for (int f = 0; f < fc; f++)
							sum += w[f](fs - 1 - k, fs - 1 - l, c) * dout(i0, j0, f);
					}
				}

				dx(i, j, c) = sum;
			}
		}
	}
}

void ConvolutionLayer::UpdateWeights(double learningRate) {
	for (int i = 0; i < fc; i++){
		b[i] -= learningRate * db[i];
		db[i] = 0;
		
		for (int j = 0; j < w[i].Total(); j++){
			w[i][j] -= learningRate * dw[i][j];
			dw[i][j] = 0;
		}
	}
}

void ConvolutionLayer::Save(ofstream &f) {
	f << "convolution " << fc << " " << fs << " " << padding << endl;

	for (int i = 0; i < fc; i++){
		for (int j = 0; j < w[i].Total(); j++)
			f << w[i][j] << " ";

		f << b[i] << endl; 
	} 
}

void ConvolutionLayer::Load(ifstream &f){
	for (int i = 0; i < fc; i++){
		for (int j = 0; j < w[i].Total(); j++)
			f >> w[i][j];

		f >> b[i]; 
	} 
}

void ConvolutionLayer::Summary() const {
	cout << "|" << setw(22) << "convolution layer|" << setw(15) << inputSize << "|" << setw(16) << outputSize << "|" << setw(14) << fc * (fs * fs * fs + 1) << "| padding: " << padding << ", " << fc << " filters [" << fs << "x" << fs << "x" << fd << "]" << endl;
}