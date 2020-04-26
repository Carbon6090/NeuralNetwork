#include <iostream>
#include "Utils/DataReader.hpp"
#include "Network.hpp"

using namespace std;

double Test(Network &network, const Data &data){
	int sum = 0;

	for (int i = 0; i < data.y.size(); i++){
		if (data.y[i].Argmax() == network.Forward(data.x[i]).Argmax())
			sum++;
	}

	return (double)sum / (double)data.y.size();
}

int main(){
	int n = 10;
	double learningRate = 0.02;
	int epochs = 500;
	int testPeriod = 1;

	DataReader reader("dataset/mnist.txt");
	Data dataTrain = reader.ReadData("dataset/mnist_train.csv");
	Data dataTest = reader.ReadData("dataset/mnist_test.csv");

	for (int i = 0; i < dataTrain.x.size(); i++)
		for (int j = 0; j < dataTrain.x[i].Total(); j++)
			dataTrain.x[i][j] /= 255.0;

	for (int i = 0; i < dataTest.x.size(); i++)
		for (int j = 0; j < dataTest.x[i].Total(); j++)
			dataTest.x[i][j] /= 255.0;

	//Network network("Models/mnist_0.987100.txt");
	Network network(reader.GetSize());
	network.AddLayer("convolution 8 3 0");
	network.AddLayer("activation relu");
	network.AddLayer("maxpooling 2");
	network.AddLayer("dropout 0.4");

	network.AddLayer("convolution 16 3 0");
	network.AddLayer("activation relu");
	network.AddLayer("maxpooling 2");
	network.AddLayer("dropout 0.4");

	network.AddLayer("fc 128");
	network.AddLayer("activation relu");
	network.AddLayer("dropout 0.2");
	network.AddLayer("fc 10");

	network.AddLayer("softmax");

	network.Summary();

	cout << "Init Accuracy" << Test(network, dataTest) << endl;

	double maxAccuracy = 0;

	for (int i = 0; i < epochs / testPeriod; i++){
		network.Train(dataTrain, learningRate, testPeriod, 1, CrossEntropy);

		double testAcc = Test(network, dataTest);
		double trainAcc = Test(network, dataTrain);

		cout << "Train accuracy" << trainAcc << endl;
		cout << "Test accuracy" << testAcc << endl << endl;

		if (testAcc > maxAccuracy){
			network.Save("Models/mnist_" + to_string(testAcc) + ".txt");
			maxAccuracy = testAcc;
		}
	}
}


/*
	conv k 3x3 - relu - conv k 3x3 - relu - pool2 - dropout 0.4 - conv 2k 3x3 - relu - conv 2k 3x3 - relu - pool - dropout 0.4 - fc n - dropout 0.2 - fc 10 - softmax
	conv k 3x3 - relu - pool2 - dropout 0.4 - conv 2k 3x3 - relu - pool - dropout 0.4 - fc n - dropout 0.2 - fc 10 - softmax

	conv 16 3x3 - relu - conv 16 3x3 - relu - pool 2 dropout 0.4 - conv 32 3x3 - relu - conv 32 3x3 - relu - pool 2 dropout 0.4 - fc 128 - dropout 0.2 - fc 10 - softmax
*/