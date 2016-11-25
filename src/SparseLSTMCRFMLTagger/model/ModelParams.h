#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	LookupTable words;
	LookupTable chars;

	SparseParams _sparse_layer;
	UniParams _tanhchar_params;
	GatedPoolParam _gate_pool_param;
	LSTM1Params _left_lstm_param;
	LSTM1Params _right_lstm_param;
	UniParams _olayer_linear;

	Alphabet _linear_feature;
	Alphabet _word_alpha;
	Alphabet _char_alpha;
	Alphabet _label_alpha;
public:
	CRFMLLoss _loss;
public:
	bool initial(HyperParams& hyper_params, AlignedMemoryPool* mem = NULL){
		if (_label_alpha.size() <= 0 || _word_alpha.size() <= 0)
			return false;
		hyper_params.labelSize = _label_alpha.size();
		hyper_params.linearFeatSize = _linear_feature.size();
		hyper_params.wordDim = words.nDim;
		hyper_params.charDim = chars.nDim;
		hyper_params.charWindow = hyper_params.charContext * 2 + 1;
		hyper_params.wordWindow = hyper_params.wordContext * 2 + 1;
		hyper_params.inputSize = hyper_params.rnnHiddenSize * 2 * hyper_params.wordWindow;
		_sparse_layer.initial(&_linear_feature, hyper_params.labelSize);
		_tanhchar_params.initial(hyper_params.charHiddenSize, hyper_params.charWindow * hyper_params.charDim, true, mem);
		_gate_pool_param.initial(hyper_params.hiddenSize, hyper_params.charHiddenSize, mem);
		_left_lstm_param.initial(hyper_params.rnnHiddenSize, hyper_params.hiddenSize + hyper_params.wordDim, mem);
		_right_lstm_param.initial(hyper_params.rnnHiddenSize, hyper_params.hiddenSize + hyper_params.wordDim, mem);
		_olayer_linear.initial(hyper_params.labelSize, hyper_params.inputSize, false, mem);
		_loss.initial(hyper_params.labelSize);
		return true;
	}

	void exportModelParams(ModelUpdate& ada){
		words.exportAdaParams(ada);
		chars.exportAdaParams(ada);
		_sparse_layer.exportAdaParams(ada);
		_tanhchar_params.exportAdaParams(ada);
		_gate_pool_param.exportAdaParams(ada);
		_left_lstm_param.exportAdaParams(ada);
		_right_lstm_param.exportAdaParams(ada);
		_olayer_linear.exportAdaParams(ada);
		_loss.exportAdaParams(ada);
	}

	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&_loss.T, "_loss.T");
		checkgrad.add(&_olayer_linear.W, "_olayer_linear.W");
		checkgrad.add(&_tanhchar_params.W, "_tanhchar_params.W");
		checkgrad.add(&_tanhchar_params.b, "_tanhchar_params.b");
		checkgrad.add(&_gate_pool_param._uni_gate_param.W, "_gate_pool_param._uni_gate_param.W");
		checkgrad.add(&_gate_pool_param._uni_gate_param.b, "_gate_pool_param._uni_gate_param.b");
		checkgrad.add(&words.E, "words.E");
	}

	void saveModel(){
	}
	void loadModel(const string& infile){
	}
};

#endif  /*SRC_ModelParams_H_*/