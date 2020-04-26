#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "Image.hpp"

using namespace std;

struct TensorSize{
	int height;
	int width;
	int depth;	
};

class Tensor {
	TensorSize size;
	vector<double> values;
public:
	Tensor(int size);
	Tensor(TensorSize size);
	Tensor(int height, int width, int depth);

	int Total() const;
	int Argmax() const;

	double& operator[](int index);
	const double& operator[](int index) const;
	
	double& operator()(int i, int j, int d);
	const double& operator()(int i, int j, int d) const;

	void SaveAsImage(const string &path);
};

Tensor::Tensor(int s) {
	size.height = 1;
	size.width = 1;
	size.depth = s;
	values = vector<double>(size.depth, 0);
}

Tensor::Tensor(TensorSize size) {
	this->size = size;
	values = vector<double>(size.height * size.width * size.depth, 0);
}

Tensor::Tensor(int height, int width, int depth){
	this->size.height = height;
	this->size.width = width;
	this->size.depth = depth;
	values = vector<double>(size.height * size.width * size.depth, 0);
}

int Tensor::Total() const {
	return values.size();
}

int Tensor::Argmax() const {
	int imax = 0;
	
	for (int i = 1; i < values.size(); i++)
		if (values[i] > values[imax])
			imax = i;

	return imax;
}

double& Tensor::operator[](int index) {
	return values[index];
}

const double& Tensor::operator[](int index) const {
	return values[index];
}
	
double& Tensor::operator()(int i, int j, int d) {
	return values[(i * size.width + j) * size.depth + d];
}

const double& Tensor::operator()(int i, int j, int d) const {
	return values[(i * size.width + j) * size.depth + d];
}

void Tensor::SaveAsImage(const string &path) {
	Image image(size.width, size.height);

	for (int i = 0; i < size.height; i++){
		for (int j = 0; j < size.width; j++){
			if (size.depth == 1){
				int v = (*this)(i, j, 0) * 255;
				image.SetPixel(j, i, v, v, v);
			}
			else{
				int r = (*this)(i, j, 0) * 255;
				int g = (*this)(i, j, 1) * 255;
				int b = (*this)(i, j, 2) * 255;
				image.SetPixel(j, i, r, g, b);
			}
		}
	}

	image.Save(path);
}

ostream& operator<<(ostream& os, const TensorSize& size) {
	return os << (to_string(size.width) + "x" + to_string(size.height) + "x" + to_string(size.depth));
}