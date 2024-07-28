const double learningRate = 0.01;

#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <bits/stdc++.h>
#include <random>
#include <vector>
#include <algorithm>
#include "utils.cpp"
#include "neuralnetwork.cpp"
#include "conv.cpp"
#include "maxpool.cpp"

int main(){
    std::vector<std::pair<std::vector<std::vector<double>>, std::vector<double>>> data = loadDataset(300);
    //std::cout << " num: "<< data[0].second[0] << std::endl;
    const int epoch = 2000; 
    NeuralNetwork nn;
    //cnn.initialise_weights();
    Convolution conv;
    MaxPool maxPoolLayer;
    conv.init(5, 5, 28, 28);
    maxPoolLayer.init();
    auto rng = std::default_random_engine {};
    //double lossNN = 0;
    for (int i = 0; i < epoch; i++){
        double lossTotal = 0;
        for (int j = 0; j < 300; j++){
            std::vector<std::vector<double>> image = data[j].first;
            
            std::vector<std::vector<std::vector<double>>> conv1 = conv.forward(image);
            std::vector<std::vector<std::vector<double>>> maxPoll = maxPoolLayer.forward(conv1);
            std::vector<double> pred = nn.feedforward(maxPoll, data[j].second);
            
            std::vector<double> dense_backprop = nn.backpropogation(pred, data[j].second);
            std::vector<std::vector<std::vector<double>>> maxPool_backprop = maxPoolLayer.backward(dense_backprop);
            conv.backward(image, maxPool_backprop);
            
            nn.update_weights();
            conv.update_weights();
            lossTotal += nn.getLoss();
        }
        std::shuffle(std::begin(data), std::end(data), rng);
        std::cout << "----------------" << std::endl; 
        std::cout << "Epoch: " << i << std::endl;  
        std::cout << "Loss: " << lossTotal << std::endl;

    	int val_len = 600;
	    int cor = 0;

        std::cout << "Start Testing." << std::endl;
	    for (int j = 0; j < val_len; j++) {
		    std::vector<std::vector<double>> image = data[j].first;
            
            std::vector<std::vector<std::vector<double>>> conv1 = conv.forward(image);
            std::vector<std::vector<std::vector<double>>> maxPoll = maxPoolLayer.forward(conv1);
            std::vector<double> pred = nn.feedforward(maxPoll, data[j].second);
		    if (arg_max(pred) == arg_max(data[j].second)) cor++;
	    }
	    float accu = double(cor) / val_len;
	    std::cout << "Accuracy: " << accu << std::endl;        

    }

    /*for (int i = 0; i < pred.size(); i++){
        std::cout << "pred: " << pred[i] << std::endl;
    }*/
    
    //for (int i = 0; i < dense_backprop.size(); i++){
    //std::cout << "dense_backprop: " << dense_backprop.size() << std::endl;
    //std::cout << "maxPool_backprop: " << maxPool_backprop.size() << std::endl;
    //}
    //std::cout << "loss: " <<  << std::endl;
    //std::cout << "derivative: " << nn.backpropogation(nn.feedforward(maxPoll, numLabels[7]), numLabels[7]) << std::endl;
}

        /*std::cout << "label 7: ";
        std::vector<std::vector<double>> lablSeven = loadImage("dataset/minst/test/7/111.jpg");
        std::vector<std::vector<std::vector<double>>> conv1Seven = conv.forward(lablSeven);
        std::vector<std::vector<std::vector<double>>> maxPollSeven = maxPoolLayer.forward(conv1Seven);
        std::vector<double> predSeven = nn.feedforward(maxPollSeven, numLabels[7]);
        for (int j= 0; j < predSeven.size(); j++){
            //std::cout << "Epoch: " << i << std::endl;
            std::cout << j << ": " << predSeven[j] << " |";
        }
        std::cout << "\n"; 
        std::cout << "label 4: ";  
        std::vector<std::vector<double>> lablFour = loadImage("dataset/minst/test/4/1010.jpg");
        std::vector<std::vector<std::vector<double>>> conv1Four = conv.forward(lablFour);
        std::vector<std::vector<std::vector<double>>> maxPollFour = maxPoolLayer.forward(conv1Four);
        std::vector<double> predFour = nn.feedforward(maxPollFour, numLabels[4]);
        for (int j= 0; j < predFour.size(); j++){
            //std::cout << "Epoch: " << i << std::endl;
            std::cout << j << ": " << predFour[j] << " |";
        }
        std::cout << "\n"; */
