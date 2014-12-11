//[[Rcpp::depends(RcppArmadillo)]]
#include "RcppArmadillo.h"
using namespace arma;

struct Imp_DataPoint;
struct Imp_Dataset;
struct Imp_OnlineOutput;
struct Imp_Experiment;

struct Imp_DataPoint {
//@members
  mat x;
  double y;
};

struct Imp_Dataset
{
//@members
  mat X;
  mat Y;
};

struct Imp_OnlineOutput{
  //Construct Imp_OnlineOutput compatible with
  //the shape of data
  Imp_OnlineOutput(Imp_Dataset data){}
  
  Imp_OnlineOutput(){}
//@members
  mat estimates;
//@methods
  mat last_estimate(){
  	return mat();
  }
};

struct Imp_Experiment {
//@members
  mat theta.star;
  unsigned p;
  unsigned n_iters;
  //mat cov_mat;
  //mat fisher_info_mat;
  string model_name;
//@methods
  // Imp_Dataset sample_dataset(){
  // 	return Imp_Dataset();
  // }
  double score_function(){
  	return 0;
  }
  double h_transfer(){
  	return 0;
  }
  double h_first_derivative(){
  	return 0;
  }
  double h_second_derivative(){
  	return 0;
  }
};
