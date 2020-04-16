#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "Data.hpp"
#include "Tensor.hpp"

using namespace std;

class DataReader{
	TensorSize size;
	vector<string> labels;
	vector<string> SplitLine(string line, char separator);
public:
	DataReader(const string &path);
	TensorSize GetSize() const;
	Tensor PixelsToVector(const vector<string> &values) const;
	Tensor LabelToVector(const string &label) const;
	Data ReadData(const string &path);
};

DataReader::DataReader(const string &path){
	ifstream f(path);

	if (!f)
		throw string("Unable to open file '") + path + "'";
	
	string line;

	getline(f, line);
	vector<string> s = SplitLine(line, ' ');
	
	size.width = stoi(s[0]);
	size.height = stoi(s[1]);
	size.depth = stoi(s[2]);

	getline(f, line);
	labels = SplitLine(line, ' ');
}

TensorSize DataReader::GetSize() const{
	return size;
}

vector<string> DataReader::SplitLine(string line, char separator){
	vector<string> s;
	string l = "";
	
	for (int i = 0; i < line.length(); i++){
		if (line[i] == separator){
			s.push_back(l);
			l = "";
		}
		else
		 	l += line[i];
	}

	if (l != "")
		s.push_back(l);

	return s;
}

Tensor DataReader::PixelsToVector(const vector<string> &values) const {
	Tensor res(size);
	int index = 1;
	for (int i = 0; i < size.height; i++)
		for (int j = 0; j < size.width; j++)
			for (int k = 0; k < size.depth; k++)
				res(i, j, k) = atof(values[index++].c_str());

	return res;
}

Tensor DataReader::LabelToVector(const string &label) const {
	Tensor res(labels.size());

	for (int i = 0; i < labels.size(); i++) {
		if (labels[i] == label){
			res[i] = 1;
			return res;
		}
	}

	throw runtime_error("Invalid label");
}

Data DataReader::ReadData(const string &path) {
	Data data;
	ifstream f(path);

	string line;
	getline(f, line);

	while(getline(f, line)){
		vector<string> s = SplitLine(line, ',');

		if (s.size() - 1 != size.width * size.height * size.depth)
			throw runtime_error("invalid pixels' size");

		data.x.push_back(PixelsToVector(s));
		data.y.push_back(LabelToVector(s[0]));
	}

	return data;
}