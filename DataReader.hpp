#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "Network.hpp"

using namespace std;

class DataReader{
	int width;
	int height;
	vector<string> labels;
public:
	DataReader(const string &path);
	vector<string> SplitLine(string line, char separator);
	vector<double> PixelsToVector(const vector<string> &values) const;
	vector<double> LabelToVector(const string &label) const;
	Data ReadData(const string &path);
};

DataReader::DataReader(const string &path){
	ifstream f(path);

	if (!f)
		throw string("Unable to open file '") + path + "'";
	
	string line;

	getline(f, line);
	vector<string> s = SplitLine(line, ' ');
	
	width = stoi(s[0]);
	height = stoi(s[1]);

	getline(f, line);
	labels = SplitLine(line, ' ');
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

vector<double> DataReader::PixelsToVector(const vector<string> &values) const {
	vector<double> res(width * height);

	for (int i = 1; i < values.size(); i++)
		res[i - 1] = atof(values[i].c_str());

	return res;
}

vector<double> DataReader::LabelToVector(const string &label) const {
	vector<double> res(labels.size(), 0);

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

		if (s.size() - 1 != width * height)
			throw runtime_error("invalid pixels' size");

		data.x.push_back(PixelsToVector(s));
		data.y.push_back(LabelToVector(s[0]));
	}

	return data;
}