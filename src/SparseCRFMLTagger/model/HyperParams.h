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

	//auto generated
	int linearFeatSize;
	int inputSize;
	int labelSize;

	// for optimization
	dtype nnRegular, adaAlpha, adaEps;

public:
	HyperParams(){
		bAssigned = false;
	}

	void setReqared(Options& opt){
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