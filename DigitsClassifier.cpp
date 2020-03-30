#include <iostream>
#include "DataReader.hpp"
#include "Network.hpp"

using namespace std;

int Argmax(const vector<double> &x){
	int imax = 0;
	
	for (int i = 1; i < x.size(); i++)
		if (x[i] > x[imax])
			imax = i;

	return imax;
}

double Test(Network &network, const Data &data){
	int sum = 0;
	for (int i = 0; i < data.y.size(); i++){
		if (Argmax(data.y[i]) == Argmax(network.Forward(data.x[i])))
			sum++;
	}

	return (double)sum / (double)data.y.size();
}

int main(){
	int n = 10;
	double learningRate = 0.1;
	int epochs = 500;
	int testPeriod = 5;

	DataReader data("dataset/mnist.txt");
	Data dataTrain = data.ReadData("dataset/mnist_train.csv");
	Data dataTest = data.ReadData("dataset/mnist_test.csv");

	for (int i = 0; i < dataTrain.x.size(); i++)
		for (int j = 0; j < dataTrain.x[i].size(); j++)
			dataTrain.x[i][j] /= 255.0;

	for (int i = 0; i < dataTest.x.size(); i++)
		for (int j = 0; j < dataTest.x[i].size(); j++)
			dataTest.x[i][j] /= 255.0;

	Network network(784);
	network.AddLayer(10, "sigmoid");

	cout << "Init Accuracy" << Test(network, dataTest) << endl;

	for (int i = 0; i < epochs / testPeriod; i++){
		network.Train(dataTrain.x, dataTrain.y, learningRate, testPeriod, 1);

		cout << "Train accuracy" << Test(network, dataTrain) << endl;
		cout << "Test accuracy" << Test(network, dataTest) << endl << endl;
	}
}