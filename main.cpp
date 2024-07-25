#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <bits/stdc++.h>
#include <random>
#include <vector>
#include "utils.cpp"
#include "neuralnetwork.cpp"
#include "conv.cpp"

int main(){
    //loadDataset();
    ConvolutionalNeuralNetwork cnn;
    //cnn.initialise_weights();
    Convolution conv; 
    conv.init(5, 5, 28, 28);
    std::vector<std::vector<double>> image = loadImage("dataset/minst/test/7/0.jpg");

    std::vector<std::vector<std::vector<double>>> conv1 = conv.forward(image);
    conv.backward(image, conv1);
}
