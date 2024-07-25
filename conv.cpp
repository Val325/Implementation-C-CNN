class Convolution {       
    private:
        int conv_number;
        int filter_size;
        int sizeH;
        int sizeW;
        
        std::vector<std::vector<std::vector<double>>> conv_w;
        std::vector<std::vector<std::vector<double>>> conv_b; 
        
        std::vector<std::vector<std::vector<double>>> conv_layer;
        std::vector<std::vector<std::vector<double>>> sig_layer;

        std::vector<std::vector<std::vector<double>>> dw_conv;
        std::vector<std::vector<std::vector<double>>> db_conv;          
        //double dw_conv[5][5][5];
        //double db_conv[5][28][28];
        
        //double conv_layer[5][28][28];
        //double sig_layer[5][28][28];
        //double conv_w[5][5][5];
        //double conv_b[5][28][28];
    public:
    Convolution(){

    }
    void init(int conv_n, int filter_s, int img_h, int img_w){
        conv_number = conv_n; // 5
        filter_size = filter_s; // 5
        sizeH = img_h; //28
        sizeW = img_w; //28
        
        conv_w.resize(conv_number); // 5
        conv_b.resize(conv_number);
        conv_layer.resize(conv_number);
        sig_layer.resize(conv_number);
        db_conv.resize(conv_number);
        dw_conv.resize(conv_number);
	    for (int i = 0; i < 5; i++) {
            conv_w[i].resize(filter_size);
            dw_conv[i].resize(filter_size);
            for (int j = 0; j < 5; j++){
                conv_w[i][j].resize(filter_size);
                dw_conv[i][j].resize(filter_size);

            }

            conv_b[i].resize(sizeH);
            conv_layer[i].resize(sizeH);
            sig_layer[i].resize(sizeH);
            db_conv[i].resize(sizeH);
		    for (int j = 0; j < sizeH; j++) {
                
                conv_b[i][j].resize(sizeW);
                conv_layer[i][j].resize(sizeW);
                sig_layer[i][j].resize(sizeW);
                db_conv[i][j].resize(sizeW);

			    for (int k = 0; k < sizeW; k++) {
				    if (j < 5 && k < 5) {
					    conv_w[i][j][k] = 2 * double(rand()) / RAND_MAX - 1; // Random double value between -1 and 1
				    }
				    conv_b[i][j][k] = 2 * double(rand()) / RAND_MAX - 1; // Random double value between -1 and 1
			    }
		    }
	    }


    }
    std::vector<std::vector<std::vector<double>>> forward(std::vector<std::vector<double>> img){
        /*std::cout << "-------------------------------" << std::endl;
        std::cout << "|          FORWARD            |" << std::endl;
        std::cout << "-------------------------------" << std::endl;
        */
   	    for (int filter_dim = 0; filter_dim < conv_number; filter_dim++) {
            //std::cout << "-------------------------------" << std::endl;
		    for (int i = 0; i < sizeH; i++) {
			    for (int j = 0; j < sizeW; j++) {
				    //max_pooling[filter_dim][i][j] = 0;

				    conv_layer[filter_dim][i][j] = 0;
				    sig_layer[filter_dim][i][j] = 0;
				    for (int k = 0; k < filter_size; k++) {
					    for (int l = 0; l < filter_size; l++) {
						    //conv_layer[filter_dim][i][j] = img[-2 + i + k][-2 + j + l] * conv_w[filter_dim][k][l];
                            if (((-2 + i + k) > 0) 
                            && ((-2 + j + l) > 0) 
                            && ((-2 + i + k) < 28) 
                            && ((-2 + j + l) < 28)) conv_layer[filter_dim][i][j] = img[-2 + i + k][-2 + j + l] * conv_w[filter_dim][k][l];
					    }
				    }
				    sig_layer[filter_dim][i][j] = sigmoid(conv_layer[filter_dim][i][j] + conv_b[filter_dim][i][j]);
                    //std::cout << sig_layer[filter_dim][i][j] << " ";  
			    }
                //std::cout << "\n";
		    }
            //std::cout << "-------------------------------" << std::endl;

	    }

        return sig_layer; 
    }
    void backward(std::vector<std::vector<double>> img, std::vector<std::vector<std::vector<double>>> dw_perv){
        /*std::cout << "-------------------------------" << std::endl;
        std::cout << "|          BACKWARD           |" << std::endl;
        std::cout << "-------------------------------" << std::endl;
   	    */
        // Calc Conv Bias Changes
	    for (int filter_dim = 0; filter_dim < dw_perv.size(); filter_dim++) {
		    for (int i = 0; i < dw_perv[0].size(); i++) {
			    for (int j = 0; j < dw_perv[0][0].size(); j++) {
				    db_conv[filter_dim][i][j] = dw_perv[filter_dim][i][j];
			    }
		    }
	    }

	    // Set Conv Layer Weight changes to 0
	    for (int filter_dim = 0; filter_dim < dw_conv.size(); filter_dim++) {
		    for (int i = 0; i < dw_conv[0].size(); i++) {
			    for (int j = 0; j < dw_conv[0][0].size(); j++) {
				    dw_conv[filter_dim][i][j] = 0;
			    }
		    }
	    }

	    // Calculate Weight Changes for Conv Layer
	    for (int filter_dim = 0; filter_dim < dw_perv.size(); filter_dim++) {
            //std::cout << "-------------------------------" << std::endl;
		    for (int i = 0; i < dw_perv[0].size(); i++) {
			    for (int j = 0; j < dw_perv[0][0].size(); j++) {
				    double cur_val = dw_perv[filter_dim][i][j];
				    for (int k = 0; k < 5; k++) {
					    for (int l = 0; l < 5; l++) {
						    if (((-2 + i + k) > 0) 
                            && ((-2 + j + l) > 0) 
                            && ((-2 + i + k) < 28) 
                            && ((-2 + j + l) < 28)){ 
                                dw_conv[filter_dim][k][l] += img[i - 2 + k][j - 2 + l] * cur_val;
                                //std::cout << dw_conv[filter_dim][k][l] << " ";  
                            }
					    }
                        //std::cout << "\n";
				    }
			    }
		    }
            //std::cout << "-------------------------------" << std::endl;
	    } 
    }
};
