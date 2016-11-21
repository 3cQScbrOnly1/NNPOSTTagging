#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{
	// must assign
	dtype dropProb;
	int rnnHiddenSize;
	int charHiddenSize;
	int hiddenSize;
	int charContext;
	int wordContext;

	//auto generated
	int linearFeatSize;
	int charWindow;
	int wordWindow;
	int wordDim;
	int charDim;
	int inputSize;
	int labelSize;

	// for optimization
	dtype nnRegular, adaAlpha, adaEps;

public:
	HyperParams(){
		bAssigned = false;
	}

	void setReqared(Options& opt){
		rnnHiddenSize = opt.rnnHiddenSize;
		charContext = opt.charcontext;
		wordContext = opt.wordcontext;
		charHiddenSize = opt.charhiddenSize;
		hiddenSize = opt.hiddenSize;
		dropProb = opt.dropProb;
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bVaild(){
		return bAssigned;
	}

	void print(){}

private:
	bool bAssigned;
};
#endif /*SRC_HyperParams_H_*/