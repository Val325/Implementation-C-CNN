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

    public:
    NeuralNetwork(){
        initialise_weights();
    }
    void initialise_weights() {
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
    void feedforward(std::vector<std::vector<std::vector<double>>> max_layer){
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
            std::cout << "prob : " << dense_softmax[i] << " num: " << i << std::endl; 
	    }
    }
    void backpropogation(){

    }
};
