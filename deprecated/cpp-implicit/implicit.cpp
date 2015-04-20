// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
//[[Rcpp::depends(RcppArmadillo)]]
#include "RcppArmadillo.h"
#include <boost/math/common_factor.hpp>
#include <boost/math/tools/roots.hpp>
#include <math.h>
#include <string>
#include <cstddef>
// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
// This file will be compiled with C++11
// BH provides methods to use boost library
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(BH)]]



using namespace arma;

#define nullptr NULL

struct Imp_DataPoint;
struct Imp_Dataset;
struct Imp_OnlineOutput;
struct Imp_Identity;
struct Imp_Exp;
//template<typename TRANSFER>
struct Imp_Experiment;
struct Imp_Size;
struct Imp_Learning_rate;
struct Imp_transfer;


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

struct Imp_Learning_rate {
  Imp_Learning_rate():gamma(1), alpha(1), c(1), scale(1) {}
  Imp_Learning_rate(double g, double a, double cin, double s):
    gamma(g), alpha(a), c(cin), scale(s) {}
  double gamma;
  double alpha;
  double c;
  double scale;
  double operator() (unsigned t) const {
    return scale * gamma * pow(1 + alpha * gamma * t, -c);
  }
};

// Base transfer function struct
struct Imp_Transfer_Base
{
  Imp_Transfer_Base() : name_("abstract transfer") { }
  virtual ~Imp_Transfer_Base() { }

  virtual double operator()(double u) const = 0;

  virtual double first(double u) const = 0;

  virtual double second(double u) const = 0;

protected:
  Imp_Transfer_Base(std::string n) : name_(n) { }
  std::string name_;
};

//Identity transfer function
struct Imp_Identity : public Imp_Transfer_Base {
  Imp_Identity() : Imp_Transfer_Base("identity transfer") { }

  virtual double operator() (double u) const{
    return u;
  }
  virtual double first(double u) const{
    return 1;
  }
  virtual double second(double u) const{
    return 0;
  }
};

//exponent transfer function
struct Imp_Exp : public Imp_Transfer_Base {
  Imp_Exp() : Imp_Transfer_Base("exponential transfer") { }

  virtual double operator() (double u) const{
    return exp(u);
  }
  virtual double first(double u) const{
    return exp(u);
  }
  virtual double second(double u) const{
    return exp(u);
  }
};

//template<typename TRANSFER>
struct Imp_Experiment {
//@members
  unsigned p;
  unsigned n_iters;
  Imp_Learning_rate lr;
  std::string model_name;
  //TRANSFER h_transfer;
//@methods
  Imp_Experiment(std::string transfer_name) : h_transfer_(nullptr) {
    if (transfer_name == "identity") {
      h_transfer_ = new Imp_Identity;
    }
    else if (transfer_name == "exp") {
      h_transfer_ = new Imp_Exp;
    }

    if (h_transfer_ == nullptr) {
      // base transfer function type
      h_transfer_ = new Imp_Identity;
    }
  }

  Imp_Experiment() : h_transfer_(new Imp_Identity) { }

  ~Imp_Experiment() {
    delete h_transfer_;
  }

  double learning_rate(unsigned t) const{
    if (model_name == "poisson")
      return double(10)/3/t;
    else if (model_name == "normal") {
      return lr(t);
    }
    return 0;
  }

  mat score_function(const mat& theta_old, const Imp_DataPoint& datapoint) const{
    return ((datapoint.y - h_transfer(as_scalar(datapoint.x * theta_old)))*datapoint.x).t();
  }

  double h_transfer(double u) const {
    return (*h_transfer_)(u);
  }

  //YKuang
  double h_first_derivative(double u) const{
    //return h_transfer.first(u);
    return h_transfer_->first(u);
  }
  //YKuang
  double h_second_derivative(double u) const{
    //return h_transfer.second(u);
    return h_transfer_->second(u);
  }

private:
  // since this is a dangerous dynamic pointer, we may want to make it private later.
  Imp_Transfer_Base* h_transfer_;

  double sigmoid(double u) const {
    return 1. / (1. + exp(-u));
  }
};

struct Imp_Size{
  Imp_Size():nsamples(0), p(0){}
  Imp_Size(unsigned nin, unsigned pin):nsamples(nin), p(pin) {}
  unsigned nsamples;
  unsigned p;
};

// Compute score function coeff and its derivative for Implicit-SGD update
//template<typename TRANSFER>
struct Get_score_coeff{

  //Get_score_coeff(const Imp_Experiment<TRANSFER>& e, const Imp_DataPoint& d,
  Get_score_coeff(const Imp_Experiment& e, const Imp_DataPoint& d,
      const mat& t, double n) : experiment(e), datapoint(d),
    theta_old(t), normx(n) {}

  double operator() (double ksi) const{
    return datapoint.y-experiment.h_transfer(dot(theta_old, datapoint.x)
                     + normx * ksi);
  }

  double first_derivative (double ksi) const{
    return experiment.h_first_derivative(dot(theta_old, datapoint.x)
           + normx * ksi)*normx;
  }

  double second_derivative (double ksi) const{
    return experiment.h_second_derivative(dot(theta_old, datapoint.x)
             + normx * ksi)*normx*normx;
  }

  //const Imp_Experiment<TRANSFER>& experiment;
  const Imp_Experiment& experiment;
  const Imp_DataPoint& datapoint;
  const mat& theta_old;
  double normx;
};

// Root finding functor for Implicit-SGD update
//template<typename TRANSFER>
struct Implicit_fn{
  typedef boost::math::tuple<double, double, double> tuple_type;

  //Implicit_fn(double a, const Get_score_coeff<TRANSFER>& get_score): at(a), g(get_score){}
  Implicit_fn(double a, const Get_score_coeff& get_score): at(a), g(get_score){}
  tuple_type operator() (double u) const{
    double value = u - at * g(u);
    double first = 1 + at * g.first_derivative(u);
    double second = at * g.second_derivative(u);
    tuple_type result(value, first, second);
    return result;
  }
  
  double at;
  //const Get_score_coeff<TRANSFER>& g;
  const Get_score_coeff& g;
};



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
//template<typename TRANSFER>
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
//template<typename TRANSFER>
mat Imp_asgd_online_algorithm(unsigned t, Imp_OnlineOutput& online_out,
	const Imp_Dataset& data_history, const Imp_Experiment& experiment){
	return Imp_sgd_online_algorithm(t, online_out, data_history, experiment);
}

//Tlan
// return the new estimate of parameters, using implicit SGD
//template<typename TRANSFER>
mat Imp_implicit_online_algorithm(unsigned t, Imp_OnlineOutput& online_out,
	const Imp_Dataset& data_history, const Imp_Experiment& experiment){
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
      result = boost::math::tools::schroeder_iterate(implicit_fn, (lower+upper)/2, lower, upper, 14);
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
  
  std::string exp_name = Rcpp::as<std::string>(Experiment["name"]);
  std::string transfer_name = Rcpp::as<std::string>(Experiment["transfer.name"]);
  Rcpp::Rcout << exp_name << ", " << transfer_name << std::endl;

  Imp_Experiment exprm(transfer_name);

  Imp_Dataset data;
  data.X = Rcpp::as<mat>(Dataset["X"]);
  data.Y = Rcpp::as<mat>(Dataset["Y"]);
  
  std::string algo;
  algo =  Rcpp::as<std::string>(algorithm);

  exprm.model_name = Rcpp::as<std::string>(Experiment["name"]);
  exprm.n_iters = Rcpp::as<unsigned>(Experiment["niters"]);
  // use the min eigenvalue of the covariance of data as alpha in LR
  cx_vec eigval;
  cx_mat eigvec;
  eig_gen(eigval, eigvec, data.covariance());
  double lr_alpha = min(eigval).real();
  Rcpp::Rcout << "learning rate alpha: " << lr_alpha << std::endl;
  //exprm.lr = Imp_Learning_rate(LR["gamma0"], LR["alpha"], LR["c"], LR["scale"]);
  exprm.lr = Imp_Learning_rate(LR["gamma0"], lr_alpha, LR["c"], LR["scale"]);
  exprm.p = Rcpp::as<unsigned>(Experiment["p"]);
  
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
  return Rcpp::List();
}

