#pragma once
#include <vector>
#include "Tensor.hpp"

using namespace std;

struct Data{
	vector<Tensor> x;
	vector<Tensor> y;
};