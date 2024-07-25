class NeuralNetwork {       
    private:
        //double conv_w[5][5][5];
        //double conv_b[5][28][28];

        //double dense_w[980][120];
        //double dense_b[120];

        //double dense_w2[120][10];
        //double dense_b2[10];
        std::vector<std::vector<double>> dense_w;
        std::vector<double> dense_b;

        std::vector<std::vector<double>> dense_w2;
        std::vector<double> dense_b2;

    public:
    NeuralNetwork(){
        initialise_weights();
    }
    void initialise_weights() {
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
    void feedforward(){

    }
    void backpropogation(){

    }
};
