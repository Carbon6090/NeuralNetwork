#pragma once
#include <iostream>
#include "Layer.hpp"

using namespace std;

class MaxPoolingLayer : public Layer{
	int scale;
public:
	MaxPoolingLayer(TensorSize size, int scale);
	
	void Forward(const Tensor &x);
	void Backward(const Tensor &x, const Tensor &dout, bool needDx);

	void Save(ofstream &f);
	void Summary() const;
};

MaxPoolingLayer::MaxPoolingLayer(TensorSize size, int scale) : Layer(size, {size.height / scale, size.width / scale, size.depth}) {
	this->scale = scale;
}

void MaxPoolingLayer::Forward(const Tensor &x) {
	for (int k = 0; k < inputSize.depth; k++)
		for (int i = 0; i < inputSize.height; i += scale)
			for (int j = 0; j < inputSize.width; j += scale) {
				double max = x(i, j, k);
				int imax = i;
				int jmax = j; 

				for (int ii = 0; ii < scale; ii++)
					for (int jj = 0; jj < scale; jj++) {
						if (max < x(i + ii, j + jj, k)) {
							max = x(i + ii, j + jj, k);
							imax = i + ii;
							jmax = j + jj;
						}

						dx(i + ii, j + jj, k) = 0;
					}
				
				dx(imax, jmax, k) = 1;
				output(i / scale, j / scale, k) = max;
	}
}

void MaxPoolingLayer::Backward(const Tensor &x, const Tensor &dout, bool needDx) {
	for (int i = 0; i < inputSize.height; i++)
		for (int j = 0; j < inputSize.width; j++)
			for (int k = 0; k < inputSize.depth; k++)
				dx(i, j, k) *= dout(i / scale, j /scale, k);
}

void MaxPoolingLayer::Save(ofstream &f) {
	f << "maxpooling " << " " << scale << endl;
}

void MaxPoolingLayer::Summary() const {
	string name = "max pooling layer |";
	cout << "|" << setw(22) << name << setw(15) << inputSize << "|"<< setw(16) << outputSize << "|" << setw(14) << "0" << "| scale: " << scale << endl;
}