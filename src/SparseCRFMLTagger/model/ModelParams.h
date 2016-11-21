#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	SparseParams _sparse_layer;

	Alphabet _linear_feature;
	Alphabet _label_alpha;
public:
	CRFMLLoss _loss;
public:
	bool initial(HyperParams& hyper_params, AlignedMemoryPool* mem = NULL){
		if (_label_alpha.size() <= 0 || _linear_feature.size() <= 0)
			return false;
		hyper_params.labelSize = _label_alpha.size();
		hyper_params.linearFeatSize = _linear_feature.size();
		_sparse_layer.initial(&_linear_feature, hyper_params.labelSize);
		_loss.initial(hyper_params.labelSize);
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		_sparse_layer.exportAdaParams(ada);
		_loss.exportAdaParams(ada);
	}

	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&_loss.T, "_loss.T");
	}

	void saveModel(){
	}
	void loadModel(const string& infile){
	}
};

#endif  /*SRC_ModelParams_H_*/