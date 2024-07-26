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
    std::vector<double> pred = nn.feedforward(maxPoll, numLabels[7]); 
    /*for (int i = 0; i < pred.size(); i++){
        std::cout << "pred: " << pred[i] << std::endl;
    }*/
    std::vector<double> dense_backprop = nn.backpropogation(pred, numLabels[7]);
    //for (int i = 0; i < dense_backprop.size(); i++){
    std::cout << "dense_backprop: " << dense_backprop.size() << std::endl;
    std::vector<std::vector<std::vector<double>>> maxPool_backprop = maxPoolLayer.backward(dense_backprop);
    std::cout << "maxPool_backprop: " << maxPool_backprop.size() << std::endl;
    conv.backward(image, maxPool_backprop);
    //}
    //std::cout << "loss: " <<  << std::endl;
    //std::cout << "derivative: " << nn.backpropogation(nn.feedforward(maxPoll, numLabels[7]), numLabels[7]) << std::endl;
}
