#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"

struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 256;
	const static int max_char_length = 16;

public:
	// node instances
	vector<LookupNode> _word_inputs;
	vector<vector<LookupNode> > _char_inputs;
	vector<vector<UniNode> > _char_tanh_project;
	vector<WindowBuilder> _char_windows;
	vector<GatedPoolBuilder> _char_gated_pooling;

	vector<ConcatNode> _word_char_concat;
	WindowBuilder _word_window;

	LSTM1Builder _left_lstm;
	LSTM1Builder _right_lstm;
	vector<ConcatNode> _concat_bi_lstm;
	vector<SparseNode> _sparse_linear;
	vector<LinearNode> _neural_linear;
	vector<PAddNode> _output;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_size, int char_size){
		_word_inputs.resize(sent_size);
		_char_inputs.resize(sent_size);
		_char_tanh_project.resize(sent_size);
		for (int idx = 0; idx < sent_size; idx++)
		{
			_char_inputs[idx].resize(char_size);
			_char_tanh_project[idx].resize(char_size);
		}
		_char_windows.resize(sent_size);
		for (int idx = 0; idx < sent_size; idx++)
			_char_windows[idx].resize(char_size);
		_char_gated_pooling.resize(sent_size);
		for (int idx = 0; idx < sent_size; idx++)
			_char_gated_pooling[idx].resize(char_size);
		_char_tanh_project.resize(sent_size);
		_word_char_concat.resize(sent_size);
		_left_lstm.resize(sent_size);
		_right_lstm.resize(sent_size);
		_concat_bi_lstm.resize(sent_size);
		_word_window.resize(sent_size);
		_sparse_linear.resize(sent_size);
		_neural_linear.resize(sent_size);
		_output.resize(sent_size);
	}

	inline void clear(){
		_word_inputs.clear();
		_char_inputs.clear();
		_char_tanh_project.clear();
		for (int idx = 0; idx < _char_inputs.size(); idx)
			_char_inputs[idx].clear();
		_char_windows.clear();
		_word_char_concat.clear();
		_char_gated_pooling.clear();
		_char_tanh_project.clear();
		_left_lstm.clear();
		_right_lstm.clear();
		_concat_bi_lstm.clear();
		_word_window.clear();
		_sparse_linear.clear();
		_neural_linear.clear();
		_output.clear();
	}

public:
	inline void initial(ModelParams& model_params, HyperParams& hyper_params, AlignedMemoryPool* mem = NULL){
		int word_max_size = _word_inputs.size();
		int char_max_size = _char_inputs[0].size();
		for (int idx = 0; idx < word_max_size; idx++) {
			_word_inputs[idx].setParam(&model_params.words);
			_word_inputs[idx].init(hyper_params.wordDim, hyper_params.dropProb, mem);
			for (int idy = 0; idy < char_max_size; idy++) {
				_char_inputs[idx][idy].setParam(&model_params.chars);
				_char_inputs[idx][idy].init(hyper_params.charDim, hyper_params.dropProb, mem);
				_char_tanh_project[idx][idy].setParam(&model_params._tanhchar_params);
				_char_tanh_project[idx][idy].init(hyper_params.charHiddenSize, hyper_params.dropProb, mem);
			}
			_char_windows[idx].init(hyper_params.charHiddenSize, hyper_params.charContext, mem);
			_char_gated_pooling[idx].init(&model_params._gate_pool_param, mem);
			_word_char_concat[idx].init(hyper_params.wordDim + hyper_params.hiddenSize, -1, mem);
			_concat_bi_lstm[idx].init(hyper_params.rnnHiddenSize * 2, -1, mem);
			_neural_linear[idx].setParam(&model_params._olayer_linear);
			_neural_linear[idx].init(hyper_params.labelSize, -1, mem);
			_sparse_linear[idx].setParam(&model_params._sparse_layer);
			_sparse_linear[idx].init(hyper_params.labelSize, -1, mem);
			_output[idx].init(hyper_params.labelSize, -1, mem);
		}
		_word_window.init(hyper_params.rnnHiddenSize * 2, hyper_params.wordContext, mem);
		_left_lstm.init(&model_params._left_lstm_param, hyper_params.dropProb, true, mem);
		_right_lstm.init(&model_params._right_lstm_param, hyper_params.dropProb, false, mem);
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const vector<Feature>& features, bool bTrain = false){
		clearValue(bTrain); // compute is a must step for train, predict and cost computation
		int seq_size, char_size;
		max_sentence_length > features.size() ? seq_size = features.size() : seq_size = max_sentence_length;
		for (int idx = 0; idx < seq_size; idx++) {
			const Feature& feature = features[idx];
			_word_inputs[idx].forward(this, feature.words[0]);
			max_char_length > feature.chars.size() ? char_size = feature.chars.size() : char_size = max_char_length;
			for (int idy = 0; idy < char_size; idy++)
				_char_inputs[idx][idy].forward(this, feature.chars[idy]);
			_char_windows[idx].forward(this, getPNodes(_char_inputs[idx], char_size));
			for (int idy = 0; idy < char_size; idy++)
				_char_tanh_project[idx][idy].forward(this, &_char_windows[idx]._outputs[idy]);
			_char_gated_pooling[idx].forward(this, getPNodes(_char_tanh_project[idx], char_size));
			_word_char_concat[idx].forward(this, &_word_inputs[idx], &_char_gated_pooling[idx]._output);
			_sparse_linear[idx].forward(this, feature.linear_features);
		}
		_left_lstm.forward(this, getPNodes(_word_char_concat, seq_size));
		_right_lstm.forward(this, getPNodes(_word_char_concat, seq_size));
		for (int idx = 0; idx < seq_size; idx++) 
			_concat_bi_lstm[idx].forward(this, &_left_lstm._hiddens[idx], &_right_lstm._hiddens[idx]);
		_word_window.forward(this, getPNodes(_concat_bi_lstm, seq_size));
		for (int idx = 0; idx < seq_size; idx++)
			_neural_linear[idx].forward(this, &_word_window._outputs[idx]);
		for (int idx = 0; idx < seq_size; idx++)
			_output[idx].forward(this, &_sparse_linear[idx], &_neural_linear[idx]);
	}

};

#endif/*SRC_ComputionGraph_H_*/