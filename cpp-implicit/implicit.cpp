
//[[Rcpp::depends(RcppArmadillo)]]
#include "RcppArmadillo.h"
using namespace arma;

struct Imp_DataPoint;
struct Imp_Dataset;
struct Imp_OnlineOutput;
struct Imp_Experiment;
struct Imp_Size;

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
  mat theta_star;
  unsigned p;
  unsigned n_iters;
  //mat cov_mat;
  //mat fisher_info_mat;
  std::string model_name;
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

struct Imp_Size{
  unsigned nsamples;
  unsigned p;
};

// Function to test the cpp integration is working
// [[Rcpp::export]]
void hello(){
	Rcpp::Rcout<<"hello world!"<<std::endl;
  Rcpp::Rcout<<"hello world!"<<std::endl;
}


//return the nsamples and p of a dataset
//std::tuple<unsigned, unsigned> Imp_dataset_size(const Imp_Dataset& dataset){
//	return std::make_tuple(0, 0);
//}

//return the nsamples and p of a dataset
Imp_Size Imp_dataset_size(const Imp_Dataset& dataset){
 return Imp_Size();
}

//add estimate to the t column of out.estimates
// Imp_OnlineOutput& Imp_add_estimate_onlineOutput(Imp_OnlineOutput& online_out, unsigned t, const mat& estimate) {
// 	return online_out;
// }

// return the @t th estimated parameter in @online_out
mat Imp_onlineOutput_estimate(const Imp_OnlineOutput& online_out, unsigned t){
	return mat();
}

// return the @t th data point in @dataset
Imp_DataPoint Imp_get_dataset_point(const Imp_Dataset& dataset, unsigned t){
	return Imp_DataPoint();
}

// return the new estimate of parameters, using SGD
mat Imp_sgd_online_algorithm(unsigned t, Imp_OnlineOutput& online_out, 
	const Imp_Dataset& data_history, const Imp_Experiment& experiment){
	return mat();
}



// return the new estimate of parameters, using ASGD
mat Imp_asgd_online_algorithm(unsigned t, Imp_OnlineOutput& online_out, 
	const Imp_Dataset& data_history, const Imp_Experiment& experiment){
	return mat();
}

// return the new estimate of parameters, using implicit SGD
mat Imp_implicit_online_algorithm(unsigned t, Imp_OnlineOutput& online_out, 
	const Imp_Dataset& data_history, const Imp_Experiment& experiment){
	return mat();
}

// transform the output of average SGD
Imp_OnlineOutput& asgd_transform_output(Imp_OnlineOutput& sgd_onlineOutput){
	return sgd_onlineOutput;
}

// use the method specified by algorithm to estimate parameters
// [[Rcpp::export]]
Rcpp::List run_online_algorithm(SEXP dataset,SEXP experiment,SEXP algorithm,
	SEXP verbose){
	return Rcpp::List();
}



