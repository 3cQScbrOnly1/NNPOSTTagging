#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"

struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 256;

public:
	// node instances
	vector<SparseNode> _output;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_size){
		_output.resize(sent_size);
	}

	inline void clear(){
		_output.clear();
	}

public:
	inline void initial(ModelParams& model_params, HyperParams& hyper_params, AlignedMemoryPool* mem = NULL){
		int maxsize = _output.size();
		for (int idx = 0; idx < maxsize; idx++) {
			_output[idx].setParam(&model_params._sparse_layer);
			_output[idx].init(hyper_params.labelSize, hyper_params.dropProb, mem);
		}
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		clearValue(bTrain); // compute is a must step for train, predict and cost computation
		int seq_size = features.size();
		for (int idx = 0; idx < seq_size; idx++){
			const Feature& feature= features[idx];
			_output[idx].forward(this, feature.linear_features);
		}
	}

};

#endif/*SRC_ComputionGraph_H_*/