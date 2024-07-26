#include "CImg.h"

std::vector<std::vector<double>> numLabels = {
    {1,0,0,0,0,0,0,0,0,0}, // 0
    {0,1,0,0,0,0,0,0,0,0}, // 1
    {0,0,1,0,0,0,0,0,0,0}, // 2
    {0,0,0,1,0,0,0,0,0,0}, // 3
    {0,0,0,0,1,0,0,0,0,0}, // 4
    {0,0,0,0,0,1,0,0,0,0}, // 5
    {0,0,0,0,0,0,1,0,0,0}, // 6
    {0,0,0,0,0,0,0,1,0,0}, // 7
    {0,0,0,0,0,0,0,0,1,0}, // 8
    {0,0,0,0,0,0,0,0,0,1}  // 9
};

template <class T>
T FindMin(std::vector<std::vector<T>> array){
    T minNumber = INT_MAX;
    for (int i = 0; i < array.size(); ++i){
        for (int j = 0; j < array[0].size(); ++j){
                if (array[i][j] < minNumber) {
                    minNumber = array[i][j];
                }
            }
        }
    
    return minNumber;
}
template <class T>
T FindMax(std::vector<std::vector<T>> array){
    T maxNumber = INT_MIN;
    for (int i = 0; i < array.size(); ++i)              // rows
    {   
        for (int j = 0; j < array[0].size(); ++j){
            if (array[i][j] > maxNumber) {
                maxNumber = array[i][j];
            }
        }
    }
    return maxNumber;
}

template <class T>
std::vector<std::vector<T>> NormalizeImage(std::vector<std::vector<T>> image, T span, T min, T max){
    int sizeX = image.size(); 
    int sizeY = image[0].size();
    std::vector<std::vector<T>> output(sizeX, std::vector<T>(sizeY, 0)); 
    for(unsigned int i = 0; i != sizeX; ++i ) {
       for(unsigned int j = 0; j != sizeY; ++j ) {
            output[i][j] = (span * (image[i][j] - min) / (max-min));
            //std::cout << "output[i][j]: " << output[i][j] << std::endl;
        } 
    }
    return output; 
}

std::vector<std::vector<double>> loadImage(std::string filepath){
    cimg_library::CImg<unsigned char> img(filepath.c_str());
    int w=img.width();
    int h=img.height();
    int c=img.spectrum();
    //std::cout << "Dimensions: " << w << "x" << h << " " << c << " channels" << std::endl;
    std::vector<std::vector<double>> Image;

    for(int y=0;y<h;y++){
       std::vector<double> xImg;
       for(int x=0;x<w;x++){
           xImg.push_back((double)img(x,y));
           //std::cout << y << "," << x << " " << (double)img(x,y) << std::endl;
       }
       Image.push_back(xImg);
       xImg.clear();
    }
    double min = FindMin(Image);
    double max = FindMax(Image);
    return NormalizeImage(Image, 1.0, min, max);
}

std::vector<std::pair<std::vector<std::vector<double>>, std::vector<double>>> loadDataset(){
        std::string path = "dataset/minst/train/";
        std::vector<std::pair<std::vector<std::vector<double>>, std::vector<double>>> dataset;

        int size = 10;
        int iterSize = 0;
        for (int i = 0; i < numLabels.size(); i++){
            std::string pathLoad = path + std::to_string(i);  
            for (const auto & entry : std::filesystem::directory_iterator(pathLoad)){
                if (iterSize >= size) break; 
                
                //std::pair<std::vector<std::vector<int>>, std::vector<double>> data;
                //data = std::make_pair(loadImage(entry.path().string()), numLabels[i]);
                //std::cout << "str: " << entry.path().string() << std::endl;
                std::vector<std::vector<double>> image = loadImage(entry.path().string());
                dataset.push_back(std::make_pair(image, numLabels[i]));
                std::cout << "amount dataset load: " << dataset.size() << std::endl;
                iterSize++;

            }
            iterSize = 0;
        }
        std::cout << "amount dataset: " << dataset.size() << std::endl;
        return dataset; 
}

double MSEloss(std::vector<double> X, std::vector<double> Y){
    int sizeOutput = X.size();
    //std::cout << "\nsizeOutput: " << sizeOutput << std::endl;
    //std::cout << "Y: " << Y << std::endl;

    double sum = 0;
    for (int i = 0; i < sizeOutput; i++) {
        //std::cout << " ((double)Y - X[i]) * ((double)Y - X[i]): " <<  ((double)Y - X[i]) * ((double)Y - X[i]) << std::endl;
        sum += (Y[i] - X[i]) * (Y[i] - X[i]);
        //std::cout << "X[i]: " <<  X[i] << std::endl;

        //std::cout << "sum: " << sum << " sizeOutput: " << sizeOutput << " (double)X[i] " << (double)X[i] << " Y: " << Y << std::endl;
    }
    sum = sum / 2*sizeOutput;
    //std::cout << "sum after (sum / sizeOutput): " << sum << std::endl;
    return sum;
}
std::vector<double> MSElossDerivative(std::vector<double> X, std::vector<double> Y){
    int sizeOutput = X.size();
    std::vector<double> output;
    for (int i = 0; i < sizeOutput; i++) {
       double deriv = ((Y[i] - (double)X[i])) / sizeOutput;
       //std::cout << "deriv: " << deriv << " sizeOutput: " << sizeOutput << " (double)X[i] " << (double)X[i] << " Y: " << Y << std::endl;       
       output.push_back(deriv);
    }
    return output;

}
double sigmoid(double x) {
	if (x > 500) x = 500;
	if (x < -500) x = -500;
	return 1 / (1 + exp(-x));
}
double d_sigmoid(double x) {
	double sig = sigmoid(x);
	return sig * (1 - sig);
}
double softmax_den(std::vector<double> x, int len) {
	double val = 0;
	for (int i = 0; i < len; i++) {
		val += exp(x[i]);
	}
	return val;
}
