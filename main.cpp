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
#include "maxpool.cpp"

int main(){
    //loadDataset();
    NeuralNetwork nn;
    //cnn.initialise_weights();
    Convolution conv;
    MaxPool maxPoolLayer;
    conv.init(5, 5, 28, 28);
    maxPoolLayer.init();
    std::vector<std::vector<double>> image = loadImage("dataset/minst/test/7/0.jpg");
    std::vector<std::vector<std::vector<double>>> conv1 = conv.forward(image);
    std::vector<std::vector<std::vector<double>>> maxPoll = maxPoolLayer.forward(conv1);
    nn.feedforward(maxPoll);
    //std::cout << maxPoll.size() << std::endl; 
}
