#include <iostream>
#include "Network.hpp"

using namespace std;

const double STEP = -50;
const double COEF = 200;

void PrintVector(vector<double> x){
	cout << "[ ";
	
	for (int i = 0; i < x.size(); i++)
		cout << (x[i] * COEF + STEP) << " ";

	cout << "]";
}

int main(){
	int n = 10;
	double learningRate = 0.4;
	int epochs = 500000;
	vector<vector<double>> x;
	vector<vector<double>> y;

	for (int i = 0; i < n; i++){
		double xi = rand() % 101 - 50;
		double yi = xi * 1.8 + 32;
		x.push_back({ (xi - STEP) / COEF });
		y.push_back({ (yi - STEP) / COEF });
	}

	Network network(1);
	network.AddLayer(2, "tanh");
	network.AddLayer(1, "sigmoid");

	network.Train(x, y, learningRate, epochs, 10000);	
	network.Print();

	for (int i = 0; i < n; i++){
		vector<double> out = network.Forward(x[i]);
		PrintVector(x[i]);
		cout << " : ";
		PrintVector(y[i]);
		cout << " != ";
		PrintVector(out);
		cout << endl; 
	}
}