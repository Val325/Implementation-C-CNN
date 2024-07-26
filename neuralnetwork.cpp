/*
double dense_input[980];
double dense_w[980][120];
double dense_b[120];
double dense_sum[120];
double dense_sigmoid[120];
double dense_w2[120][10];
double dense_b2[10];
double dense_sum2[10];
double dense_softmax[10];
*/

class NeuralNetwork {       
    private:
        //double conv_w[5][5][5];
        //double conv_b[5][28][28];

        //double dense_w[980][120];
        //double dense_b[120];

        //double dense_w2[120][10];
        //double dense_b2[10];
        std::vector<double> dense_input;
        std::vector<double> dense_sum;
        std::vector<double> dense_sigmoid;
        std::vector<double> dense_sum2;
        std::vector<double> dense_softmax;

        std::vector<std::vector<double>> dense_w;
        std::vector<double> dense_b;

        std::vector<std::vector<double>> dense_w2;
        std::vector<double> dense_b2;

        //delta backprop
        //double dw2[120][10];
        //double db2[10];
        //double dw1[980][120];
        //double db1[120]; 
        
        std::vector<std::vector<double>> dw2;
        std::vector<double> db2;

        std::vector<std::vector<double>> dw1;
        std::vector<double> db1;

    public:
    NeuralNetwork(){
        initialise_weights();
    }
    void initialise_weights() {
        dw2.resize(120);
        db2.resize(10);
        for (int i = 0; i < 120; i++){
             dw2[i].resize(10);
        }
        dw1.resize(980);
        db1.resize(120);
        for (int i = 0; i < 980; i++){
            dw1[i].resize(120);
        }

        dense_input.resize(980);
        for (int i = 0; i < 980; i++){
            dense_input[i] = 0;
        }
        
        dense_sum.resize(120);
        dense_sigmoid.resize(120);
        for (int i = 0; i < 120; i++){
            dense_sum[i] = 0;
            dense_sigmoid[i] = 0;
        }
        dense_sum2.resize(10);
        dense_softmax.resize(10);
        for (int i = 0; i < 10; i++){
            dense_sum2[i] = 0;
            dense_softmax[i] = 0;
        }

        dense_w.resize(980);
        for (int i = 0; i < 980; i++) {
            dense_w[i].resize(120);
		    for (int j = 0; j < 120; j++) {
			    dense_w[i][j] = 2 * double(rand()) / RAND_MAX - 1;
		    }
	    }
        dense_b.resize(120);
	    for (int i = 0; i < 120; i++) {
		    dense_b[i] = 2 * double(rand()) / RAND_MAX - 1;
	    }
        
        dense_w2.resize(120);
	    for (int i = 0; i < 120; i++) {
            dense_w2[i].resize(10);
		    for (int j = 0; j < 10; j++) {
			    dense_w2[i][j] = 2 * double(rand()) / RAND_MAX - 1;
		    }
	    }
        dense_b2.resize(10);
	    for (int i = 0; i < 10; i++) {
		    dense_b2[i] = 2 * double(rand()) / RAND_MAX - 1;
	    }
    }
    std::vector<double> feedforward(std::vector<std::vector<std::vector<double>>> max_layer, std::vector<double> labelY){
	    int k = 0;
	    for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
		    for (int i = 0; i < 14; i++) {
			    for (int j = 0; j < 14; j++) {
				    dense_input[k] = max_layer[filter_dim][i][j];
				    k++;
			    }
		    }
	    }

	    // Dense Layer
	    for (int i = 0; i < 120; i++) {
		    dense_sum[i] = 0;
		    dense_sigmoid[i] = 0;
		    for (int j = 0; j < 980; j++) {
			    dense_sum[i] += dense_w[j][i] * dense_input[j];
		    }
		    dense_sum[i] += dense_b[i];
		    dense_sigmoid[i] = sigmoid(dense_sum[i]);
	    }

	    // Dense Layer 2
	    for (int i = 0; i < 10; i++) {
		    dense_sum2[i] = 0;
		    for (int j = 0; j < 120; j++) {
			    dense_sum2[i] += dense_w2[j][i] * dense_sigmoid[j];
		    }
		    dense_sum2[i] += dense_b2[i];
	    }

	    // Softmax Output
	    double den = softmax_den(dense_sum2, 10);
	    for (int i = 0; i < 10; i++) {
		    dense_softmax[i] = exp(dense_sum2[i]) / den;
            //std::cout << "prob : " << dense_softmax[i] << " num: " << i << std::endl; 
	    }
        //std::cout << "loss: " << MSEloss(dense_softmax, labelY) << std::endl;
        return dense_softmax;
    }
    std::vector<double> backpropogation(std::vector<double> labelX, std::vector<double> labelY){
        std::vector<double> delta4;
        delta4.resize(10);
	    for (int i = 0; i < 10; i++) {
		    delta4[i] = labelX[i] - labelY[i]; // Derivative of Softmax + Cross entropy
		    db2[i] = delta4[i]; // Bias Changes
	    }

	    // Calculate Weight Changes for Dense Layer 2
	    for (int i = 0; i < 120; i++) {
		    for (int j = 0; j < 10; j++) {
			    dw2[i][j] = dense_sigmoid[i] * delta4[j];
		    }
	    }

	    // Delta 3
        std::vector<double> delta3;
        delta3.resize(120);
	    for (int i = 0; i < 120; i++) {
		    delta3[i] = 0;
		    for (int j = 0; j < 10; j++) {
			    delta3[i] += dense_w2[i][j] * delta4[j];
		    }
		    delta3[i] *= d_sigmoid(dense_sum[i]);
	    }
	    for (int i = 0; i < 120; i++) db1[i] = delta3[i]; // Bias Weight change

	    // Calculate Weight Changes for Dense Layer 1
	    for (int i = 0; i < 980; i++) {
		    for (int j = 0; j < 120; j++) {
			    dw1[i][j] = dense_input[i] * delta3[j];
		    }
	    }

	    // Delta2
        std::vector<double> delta2;
        delta2.resize(980);
	    for (int i = 0; i < 980; i++) {
		    delta2[i] = 0;
		    for (int j = 0; j < 120; j++){
			    delta2[i] += dense_w[i][j] * delta3[j];
		    }   
		    delta2[i] *= d_sigmoid(dense_input[i]);
	    }

        //std::cout << "derivative: " << MSElossDerivative(X, Y) << std::endl;
        return delta2; 
    }
};
