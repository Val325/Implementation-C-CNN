class MaxPool {       
    private:
    //double max_pooling[5][28][28];
    //double max_layer[5][14][14];
    //double dw_max[5][28][28];

    int sizeH;
    int sizeW;
    
    std::vector<std::vector<std::vector<double>>> max_pooling;
    std::vector<std::vector<std::vector<double>>> max_layer; 
    std::vector<std::vector<std::vector<double>>> dw_max;
    public:
    MaxPool(){

    } 
    void init(){
        sizeH = 28;
        sizeW = 28;
        max_pooling.resize(5);
        max_layer.resize(5);
        dw_max.resize(5);
        for (int i = 0; i < 5; i++) {
            max_pooling[i].resize(28);
            dw_max[i].resize(28);
            for (int j = 0; j < 28; j++) {
                max_pooling[i][j].resize(28);
                dw_max[i][j].resize(28);
            }
            max_layer[i].resize(14);
            for (int j = 0; j < 14; j++) {
                max_layer[i][j].resize(14);
            }
        }

        for (int i = 0; i < 5; i++){
            for (int j = 0; j < 28; j++) {
                for (int k = 0; k < 28; k++) {
                   max_pooling[i][j][k] = 0;
                   dw_max[i][j][k] = 0;
                }
            }
            for (int j = 0; j < 14; j++) {
                for (int k = 0; k < 14; k++) {
                    max_layer[i][j][k] = 0;
                }
            }
        }
    }
    void forward(std::vector<std::vector<std::vector<double>>> sig_layer){
	    // MAX Pooling (max_pooling, max_layer)
	    double cur_max = 0;
	    int max_i = 0, max_j = 0;
	    for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
		    for (int i = 0; i < 28; i += 2) {
			    for (int j = 0; j < 28; j += 2) {
				    max_i = i;
				    max_j = j;
				    cur_max = sig_layer[filter_dim][i][j];
				    for (int k = 0; k < 2; k++) {
					    for (int l = 0; l < 2; l++) {
						    if (sig_layer[filter_dim][i + k][j + l] > cur_max) {
							    max_i = i + k;
							    max_j = j + l;
							    cur_max = sig_layer[filter_dim][max_i][max_j];
						    }
					    }
				    }
				    max_pooling[filter_dim][max_i][max_j] = 1;
				    max_layer[filter_dim][i / 2][j / 2] = cur_max;
			    }
		    }
	    }

    }

    std::vector<std::vector<std::vector<double>>> backward(std::vector<double> perv_dense){
	    // Calc back-propagated max layer dw_max
	    int k = 0;
	    for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
		    for (int i = 0; i < 28; i += 2) {
			    for (int j = 0; j < 28; j += 2) {
				    for (int l = 0; l < 2; l++) {
					    for (int m = 0; m < 2; m++) {
						    if (max_pooling[filter_dim][i + l][j + m] == 1) dw_max[filter_dim][i][j] = perv_dense[k];
						    else dw_max[filter_dim][i][j] = 0;
					    }
				    }
				    k++;
			    }
		    }
	    }
        return dw_max;
    }
}
