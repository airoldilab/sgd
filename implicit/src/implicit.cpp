// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-


#include "RcppArmadillo.h"
#include "implicit.hpp"
#include <boost/math/common_factor.hpp>

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
// This file will be compiled with C++11
// BH provides methods to use boost library
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(BH)]]



// hello world function 
// 
//
// [[Rcpp::export]]
void hello_world() {
    Rcpp::Rcout<<"Hello!"<<"  "<<boost::math::gcd(12, 8)<<std::endl;
}


// Function to output function results for testing
// This function should cause conflicts in merging
// This function should be REMOVED after debugging
//
// [[Rcpp::export]]
arma::mat test(arma::mat input1){
  Imp_Dataset input;
  input.X = input1;
  return mat();
}

//return the nsamples and p of a dataset
Imp_Size Imp_dataset_size(const Imp_Dataset& dataset){
  Imp_Size size;
  size.nsamples = dataset.X.n_rows;
  size.p = dataset.X.n_cols;
  return size;
}

//add estimate to the t column of out.estimates
// Imp_OnlineOutput& Imp_add_estimate_onlineOutput(Imp_OnlineOutput& online_out, unsigned t, const mat& estimate) {
// 	return online_out;
// }

// return the @t th estimated parameter in @online_out
// Here, t=1 is the first estimate, which in matrix will be its 0-th col
mat Imp_onlineOutput_estimate(const Imp_OnlineOutput& online_out, unsigned t){
  if (t==0){
      return(mat(online_out.estimates.n_rows, 1, fill::zeros));
  }
  t = t-1;
  mat column = mat(online_out.estimates.col(t));
  return column;
}

// return the @t th data point in @dataset
Imp_DataPoint Imp_get_dataset_point(const Imp_Dataset& dataset, unsigned t){
  t = t - 1;
  mat xt = mat(dataset.X.row(t));
  double yt = dataset.Y(t);
  return Imp_DataPoint(xt, yt);
}

// return the new estimate of parameters, using SGD
template<typename TRANSFER>
mat Imp_sgd_online_algorithm(unsigned t, Imp_OnlineOutput& online_out,
	const Imp_Dataset& data_history, const Imp_Experiment<TRANSFER>& experiment){
  Imp_DataPoint datapoint = Imp_get_dataset_point(data_history, t);
  double at = experiment.learning_rate(t);
  mat theta_old = Imp_onlineOutput_estimate(online_out, t-1);
  mat score_t = experiment.score_function(theta_old, datapoint);
  mat theta_new = theta_old + at * score_t;
  online_out.estimates.col(t-1) = theta_new;
  return theta_new;
}

// return the new estimate of parameters, using ASGD
template<typename TRANSFER>
mat Imp_asgd_online_algorithm(unsigned t, Imp_OnlineOutput& online_out,
	const Imp_Dataset& data_history, const Imp_Experiment<TRANSFER>& experiment){
	return Imp_sgd_online_algorithm(t, online_out, data_history, experiment);
}

//Tlan
// return the new estimate of parameters, using implicit SGD
template<typename TRANSFER>
mat Imp_implicit_online_algorithm(unsigned t, Imp_OnlineOutput& online_out,
	const Imp_Dataset& data_history, const Imp_Experiment<TRANSFER>& experiment){
  Imp_DataPoint datapoint= Imp_get_dataset_point(data_history, t);
  double at = experiment.learning_rate(t);
  double normx = dot(datapoint.x, datapoint.x);
  mat theta_old = Imp_onlineOutput_estimate(online_out, t-1);

  Get_score_coeff get_score_coeff(experiment, datapoint, theta_old, normx);
  Implicit_fn implicit_fn(at, get_score_coeff);

  double rt = at * get_score_coeff(0);
  double lower = 0;
  double upper = 0;
  if (rt < 0){
      upper = 0;
      lower = rt;
  }
  else{
      upper = rt;
      lower = 0;
  }
  double result;
  if (lower != upper){
      result = boost::math::tools::newton_raphson_iterate(implicit_fn, (lower+upper)/2, lower, upper, 14);
  }
  else
    result = lower;
  mat theta_new = theta_old + result * datapoint.x.t();
  online_out.estimates.col(t-1) = theta_new;
  return theta_new;
}

// YKuang
// transform the output of average SGD
void asgd_transform_output(Imp_OnlineOutput& sgd_onlineOutput){
	mat avg_estimates(sgd_onlineOutput.estimates.n_rows, 1);
	avg_estimates = Imp_onlineOutput_estimate(sgd_onlineOutput, 1);
	for (unsigned t = 1; t < sgd_onlineOutput.estimates.n_cols; ++t) {
		avg_estimates = (1. - 1./(double)t) * avg_estimates
						+ 1./((double)t) * Imp_onlineOutput_estimate(sgd_onlineOutput, t+1);
		// t+1-th data has been averaged in @sgd_onlineOutput.estimate,
		// hence can be used to store instantly
		sgd_onlineOutput.estimates.col(t) = avg_estimates;
	}
}

// use the method specified by algorithm to estimate parameters
// [[Rcpp::export]]
Rcpp::List run_online_algorithm(SEXP dataset,SEXP experiment,SEXP algorithm,
	SEXP verbose){
  Rcpp::List Dataset(dataset);
  Rcpp::List Experiment(experiment);
  Rcpp::List LR = Experiment["lr"];
  std::string transfer_name = Rcpp::as<std::string>(Experiment["name"]);
  if (transfer_name == "identity"){
      Imp_Experiment<Imp_Identity> exprm;
      Imp_Dataset data;
      std::string algo;
      algo =  Rcpp::as<std::string>(algorithm);
      exprm.model_name = Rcpp::as<std::string>(Experiment["name"]);
      exprm.n_iters = Rcpp::as<unsigned>(Experiment["niters"]);
      exprm.lr = Imp_Learning_rate(LR["gamma0"], LR["alpha"], LR["c"], LR["scale"]);
      exprm.p = Rcpp::as<unsigned>(Experiment["p"]);
      data.X = Rcpp::as<mat>(Dataset["X"]);
      data.Y = Rcpp::as<mat>(Dataset["Y"]);
      Imp_OnlineOutput out(data);
      unsigned nsamples = Imp_dataset_size(data).nsamples;

      for(int t=1; t<=nsamples; ++t){
	  if (algo == "sgd") {
	    Imp_sgd_online_algorithm(t, out, data, exprm);
	  }
	  else if (algo == "asgd") {
	    Imp_asgd_online_algorithm(t, out, data, exprm);
	  }
	  else if (algo == "implicit"){
      Imp_implicit_online_algorithm(t, out, data, exprm);
	  }
      }
      if (algo == "asgd") {
		asgd_transform_output(out);
      }
      return Rcpp::List::create(Rcpp::Named("estimates") = out.estimates,
				  Rcpp::Named("last") = out.last_estimate());
  }
  return Rcpp::List();
}
