
//[[Rcpp::depends(RcppArmadillo)]]
#include "RcppArmadillo.h"
#include <boost/math/tools/roots.hpp>
#include <math.h>

using namespace arma;

struct Imp_DataPoint;
struct Imp_Dataset;
struct Imp_OnlineOutput;
struct Imp_Experiment;
struct Imp_Size;

arma::mat test(arma::mat input);
Imp_Size Imp_dataset_size(const Imp_Dataset& dataset);
mat Imp_onlineOutput_estimate(const Imp_OnlineOutput& online_out, unsigned t);
Imp_DataPoint Imp_get_dataset_point(const Imp_Dataset& dataset, unsigned t);
mat Imp_sgd_online_algorithm(unsigned t, Imp_OnlineOutput& online_out,
	const Imp_Dataset& data_history, const Imp_Experiment& experiment);
mat Imp_asgd_online_algorithm(unsigned t, Imp_OnlineOutput& online_out,
	const Imp_Dataset& data_history, const Imp_Experiment& experiment);
mat Imp_implicit_online_algorithm(unsigned t, Imp_OnlineOutput& online_out,
	const Imp_Dataset& data_history, const Imp_Experiment& experiment);
Imp_OnlineOutput& asgd_transform_output(Imp_OnlineOutput& sgd_onlineOutput);
Rcpp::List run_online_algorithm(SEXP dataset,SEXP experiment,SEXP algorithm,
	SEXP verbose);


struct Imp_DataPoint {
  Imp_DataPoint(): x(mat()), y(0) {}
  Imp_DataPoint(mat xin, double yin):x(xin), y(yin) {}
//@members
  mat x;
  double y;
};

struct Imp_Dataset
{
  Imp_Dataset():X(mat()), Y(mat()) {}
  Imp_Dataset(mat xin, mat yin):X(xin), Y(yin) {}
//@members
  mat X;
  mat Y;
};

struct Imp_OnlineOutput{
  //Construct Imp_OnlineOutput compatible with
  //the shape of data
  Imp_OnlineOutput(const Imp_Dataset& data):estimates(mat(data.X.n_cols, data.X.n_rows)){}
  Imp_OnlineOutput(){}
//@members
  mat estimates;
//@methods
  mat last_estimate(){
    return estimates.col(estimates.n_cols-1);
  }
};

struct Imp_Experiment {
//@members
  //mat theta_star;
  unsigned p;
  unsigned n_iters;
  //mat cov_mat;
  //mat fisher_info_mat;
  std::string model_name;
//@methods
  // Imp_Dataset sample_dataset(){
  // 	return Imp_Dataset();
  // }
  double learning_rate(unsigned t) const{
    if (model_name=="poisson")
      return double(10)/3/t;
    return 0;
  }
  mat score_function(const mat& theta_old, const Imp_DataPoint& datapoint) const{
    return ((datapoint.y - h_transfer(as_scalar(datapoint.x * theta_old)))*datapoint.x).t();
  }
  double h_transfer(double u) const{
    if (model_name=="poisson")
      return exp(u);
    return 0;
  }
  //YKuang
  double h_first_derivative() const{
  	return 0;
  }
  //YKuang
  double h_second_derivative() const{
  	return 0;
  }
};

struct Imp_Size{
  Imp_Size():nsamples(0), p(0){}
  Imp_Size(unsigned nin, unsigned pin):nsamples(nin), p(pin) {}
  unsigned nsamples;
  unsigned p;
};

// Function to test the cpp integration is working
// [[Rcpp::export]]
void hello(){
	Rcpp::Rcout<<"hello world!"<<std::endl;
  Rcpp::Rcout<<"hello world!"<<std::endl;
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
//std::tuple<unsigned, unsigned> Imp_dataset_size(const Imp_Dataset& dataset){
//	return std::make_tuple(0, 0);
//}

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
mat Imp_sgd_online_algorithm(unsigned t, Imp_OnlineOutput& online_out, 
	const Imp_Dataset& data_history, const Imp_Experiment& experiment){
  Imp_DataPoint datapoint = Imp_get_dataset_point(data_history, t);
  double at = experiment.learning_rate(t);
  mat theta_old = Imp_onlineOutput_estimate(online_out, t-1);
  mat score_t = experiment.score_function(theta_old, datapoint);
  mat theta_new = theta_old + at * score_t;
  online_out.estimates.col(t-1) = theta_new;
  return theta_new;
}



// return the new estimate of parameters, using ASGD
mat Imp_asgd_online_algorithm(unsigned t, Imp_OnlineOutput& online_out, 
	const Imp_Dataset& data_history, const Imp_Experiment& experiment){
	return mat();
}


//Tlan
// return the new estimate of parameters, using implicit SGD
mat Imp_implicit_online_algorithm(unsigned t, Imp_OnlineOutput& online_out, 
	const Imp_Dataset& data_history, const Imp_Experiment& experiment){
  Imp_DataPoint datapoint= Imp_get_dataset_point(data_history, t);
  double at = experiment.learning_rate(t);
  double normx = dot(datapoint.x, datapoint.x);
  mat theta_old = Imp_onlineOutput_estimate(online_out, t-1);

  struct Get_score_coeff{
    Get_score_coeff(const Imp_Experiment& e, const Imp_DataPoint& d,
		    const mat& t, double n):experiment(e), datapoint(d),
			theta_old(t), normx(n) {}
    double operator() (double ksi) const{
      return datapoint.y-experiment.h_transfer(dot(theta_old, datapoint.x)
      					       + normx * ksi);
    }
    const Imp_Experiment& experiment;
    const Imp_DataPoint& datapoint;
    const mat& theta_old;
    double normx;
  };

  struct Implicit_fn{
    Implicit_fn(double a, const Get_score_coeff& get_score): at(a), g(get_score){}
    double operator() (double u) const{
      return u - at * g(u);
    }
    double at;
    const Get_score_coeff& g;
  };

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
      std::pair<double, double> xit;
      boost::math::tools::eps_tolerance<double> tol(14);
      boost::uintmax_t max_iter = 1000;
      xit = boost::math::tools::toms748_solve(implicit_fn, lower, upper, tol, max_iter);
      result = (xit.first + xit.second)/2;
  }
  else
    result = lower;
  mat theta_new = theta_old + result * datapoint.x;
  online_out.estimates.col(t-1) = theta_new;
  return theta_new;
}

//YKuang
// transform the output of average SGD
Imp_OnlineOutput& asgd_transform_output(Imp_OnlineOutput& sgd_onlineOutput){
	return sgd_onlineOutput;
}

// use the method specified by algorithm to estimate parameters
// [[Rcpp::export]]
Rcpp::List run_online_algorithm(SEXP dataset,SEXP experiment,SEXP algorithm,
	SEXP verbose){
  Rcpp::List Dataset(dataset);
  Rcpp::List Experiment(experiment);
  Imp_Experiment exprm;
  Imp_Dataset data;
  std::string algo;
  algo =  Rcpp::as<std::string>(algorithm);
  exprm.model_name = Rcpp::as<std::string>(Experiment["name"]);
  exprm.n_iters = Rcpp::as<unsigned>(Experiment["niters"]);
  exprm.p = Rcpp::as<unsigned>(Experiment["p"]);
  data.X = Rcpp::as<mat>(Dataset["X"]);
  data.Y = Rcpp::as<mat>(Dataset["Y"]);
  Imp_OnlineOutput out(data);
  unsigned nsamples = Imp_dataset_size(data).nsamples;

  for(int t=1; t<=nsamples; ++t){
      if (algo == "sgd"){
	Imp_sgd_online_algorithm(t, out, data, exprm);
      }
  }

  return Rcpp::List::create(Rcpp::Named("estimates") = out.estimates,
			    Rcpp::Named("last") = out.last_estimate());
}



