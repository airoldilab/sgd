// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include "implicit.h"
#include <stdlib.h>

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
// This file will be compiled with C++11
// BH provides methods to use boost library
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(BH)]]

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
      //return(mat(online_out.estimates.n_rows, 1, fill::zeros));
    return online_out.initial;
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
	const Imp_Dataset& data_history, const Imp_Experiment& experiment, bool& good_gradient){
  static int count = 0;

  Imp_DataPoint datapoint = Imp_get_dataset_point(data_history, t);
  mat theta_old = Imp_onlineOutput_estimate(online_out, t-1);
  mat at = experiment.learning_rate(theta_old, datapoint, experiment.offset[t-1], t);
  mat score_t = experiment.score_function(theta_old, datapoint, experiment.offset[t-1]);
  if (!is_finite(score_t))
    good_gradient = false;
#if 0
  if (count < 10) {
    Rcpp::Rcout << "learning rate: \n" << at;
    Rcpp::Rcout << "Score function: \n" << score_t << std::endl;
    ++count;
  }
#endif
  mat theta_new = theta_old + mat(at * score_t);
  online_out.estimates.col(t-1) = theta_new;
  return theta_new;
}

// return the new estimate of parameters, using ASGD
//template<typename TRANSFER>
mat Imp_asgd_online_algorithm(unsigned t, Imp_OnlineOutput& online_out,
	const Imp_Dataset& data_history, const Imp_Experiment& experiment, bool& good_gradient){
	return Imp_sgd_online_algorithm(t, online_out, data_history, experiment, good_gradient);
}

//Tlan
// return the new estimate of parameters, using implicit SGD
//template<typename TRANSFER>
mat Imp_implicit_online_algorithm(unsigned t, Imp_OnlineOutput& online_out,
	const Imp_Dataset& data_history, const Imp_Experiment& experiment){
  Imp_DataPoint datapoint= Imp_get_dataset_point(data_history, t);
  mat theta_old = Imp_onlineOutput_estimate(online_out, t-1);

  mat at = experiment.learning_rate(theta_old, datapoint, experiment.offset[t-1], t);
  vec diag_lr = at.diag();
  double average_lr = 0.;
  for (unsigned i = 0; i < diag_lr.n_elem; ++i) {
    average_lr += diag_lr[i];
  }
  average_lr /= diag_lr.n_elem;

  double normx = dot(datapoint.x, datapoint.x);

  Get_score_coeff get_score_coeff(experiment, datapoint, theta_old, normx, experiment.offset[t-1]);
  Implicit_fn implicit_fn(average_lr, get_score_coeff);

  double rt = average_lr * get_score_coeff(0);
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

bool validity_check(const Imp_Dataset& data, const mat& theta, unsigned t, const Imp_Experiment& exprm){
  //check if all estimates are finite
  if (!is_finite(theta)){
    Rcpp::Rcout<<"warning: non-finite coefficients at iteration "<<t<<std::endl;
  }

  //check if eta is in the support
  double eta = exprm.offset[t-1] + as_scalar(Imp_get_dataset_point(data, t).x * theta);
  if (!exprm.valideta(eta)){
    Rcpp::Rcout<<"no valid set of coefficients has been found: please supply starting values"<<t<<std::endl;
    return false;
  }

  //check the variance of the expectation of Y
  double mu_var = exprm.variance(exprm.h_transfer(eta));
  if (!is_finite(mu_var)){
    Rcpp::Rcout<<"NA in V(mu)"<<t<<std::endl;
    return false;
  }
  if (mu_var == 0){
    Rcpp::Rcout<<"0 in V(mu)"<<t<<std::endl;
    return false;
  }
  double deviance = 0;
  mat mu;
  mat eta_mat;
  //check the deviance
  if (exprm.dev){
    eta_mat = data.X * theta + exprm.offset;
    mu = exprm.h_transfer(eta_mat);
    deviance = exprm.deviance(data.Y, mu, exprm.weights);
    if(!is_finite(deviance)){
      Rcpp::Rcout<<"Deviance is non-finite"<<std::endl;
      return false;
    }
  }
  //print if trace
  if(exprm.trace){
    if (!exprm.dev){
      eta_mat = data.X * theta + exprm.offset;
      mu = exprm.h_transfer(eta_mat);
      deviance = exprm.deviance(data.Y, mu, exprm.weights);
    }
    Rcpp::Rcout<<"Deviance = "<<deviance<<" , Iterations - "<<t<<std::endl;
  }
  return true;
}

// use the method specified by algorithm to estimate parameters
// [[Rcpp::export]]
Rcpp::List run_online_algorithm(SEXP dataset,SEXP experiment,SEXP algorithm,
	SEXP verbose){
  Rcpp::List Dataset(dataset);
  Rcpp::List Experiment(experiment);
  //Rcpp::List LR = Experiment["lr"];
  
  std::string model_name = Rcpp::as<std::string>(Experiment["name"]);
  std::string transfer_name = Rcpp::as<std::string>(Experiment["transfer.name"]);

  Imp_Experiment exprm(model_name, transfer_name);

  Imp_Dataset data;
  data.X = Rcpp::as<mat>(Dataset["X"]);
  data.Y = Rcpp::as<mat>(Dataset["Y"]);
  std::string algo;
  algo =  Rcpp::as<std::string>(algorithm);

  exprm.n_iters = Rcpp::as<unsigned>(Experiment["niters"]);
  exprm.p = Rcpp::as<unsigned>(Experiment["p"]);
  exprm.offset = Rcpp::as<mat>(Experiment["offset"]);
  exprm.weights = Rcpp::as<mat>(Experiment["weights"]);
  exprm.start = Rcpp::as<mat>(Experiment["start"]);
  exprm.convergence = Rcpp::as<bool>(Experiment["convergence"]);
  exprm.dev = Rcpp::as<bool>(Experiment["deviance"]);
  exprm.trace = Rcpp::as<bool>(Experiment["trace"]);
  exprm.epsilon = Rcpp::as<double>(Experiment["epsilon"]);
  std::string lr_type = Rcpp::as<std::string>(Experiment["lr.type"]);
  if (lr_type == "uni-dim") {
    // use the min eigenvalue of the covariance of data as alpha in LR
    cx_vec eigval;
    cx_mat eigvec;
    eig_gen(eigval, eigvec, data.covariance());
    double lr_alpha = min(eigval).real();
    exprm.init_uni_dim_learning_rate(1., lr_alpha, 2./3., 1.);
  }
  else if (lr_type == "uni-dim-eigen") {
    exprm.init_uni_dim_eigen_learning_rate();
  }
  else if (lr_type == "p-dim") {
    exprm.init_pdim_learning_rate();
  }
  else if (lr_type == "p-dim-weighted") {
    exprm.init_pdim_weighted_learning_rate(1.);
  }
  else if (lr_type == "p-dim-fisher") {
    exprm.init_pdim_fisher_learning_rate();
  }
  
  Imp_OnlineOutput out(data, exprm.start);
  unsigned nsamples = Imp_dataset_size(data).nsamples;

  //check if the number of observations is greater than the rank of X
  unsigned X_rank = rank(data.X);
  if (X_rank > nsamples){
    Rcpp::Rcout<<"X matrix has rank "<<X_rank<<", but only "
	<<nsamples<<" observation"<<std::endl;
    return Rcpp::List();
  }

  // print out info
  //Rcpp::Rcout << data;
  //Rcpp::Rcout << exprm;
  //Rcpp::Rcout << "    Method: " << algo << std::endl;

  bool good_gradient = true;
  bool good_validity = true;
  for(int t=1; t<=nsamples; ++t){
    if (algo == "sgd") {
      mat theta = Imp_sgd_online_algorithm(t, out, data, exprm, good_gradient);
      if (!good_gradient){
	Rcpp::Rcout<<"NA or infinite gradient"<<std::endl;
	return Rcpp::List();
      }
      good_validity = validity_check(data,theta, t, exprm);
      if (!good_validity)
	return Rcpp::List();
    }
    else if (algo == "asgd") {
      mat theta = Imp_asgd_online_algorithm(t, out, data, exprm, good_gradient);
      if (!good_gradient){
	Rcpp::Rcout<<"NA or infinite gradient"<<std::endl;
      	return Rcpp::List();
      }
      good_validity = validity_check(data,theta, t, exprm);
      if (!good_validity)
        Rcpp::Rcout << theta << std::endl;
      	return Rcpp::List();
    }
    else if (algo == "implicit" || algo == "a-implicit"){
      mat theta = Imp_implicit_online_algorithm(t, out, data, exprm);
      if (!is_finite(theta)){
        Rcpp::Rcout<<"warning: non-finite coefficients at iteration "<<t<<std::endl;
      }
      good_validity = validity_check(data,theta, t, exprm);
      if (!good_validity)
      	return Rcpp::List();
    }
  }
  if (algo == "asgd" || algo == "a-implicit") {
    asgd_transform_output(out);
  }

  //check the validity of eta for all observations
  mat eta;
  eta = data.X * out.last_estimate() + exprm.offset;
  mat mu;
  mu = exprm.h_transfer(eta);
  for(int i=0; i<eta.n_rows; ++i){
      if (!is_finite(eta[i])){
	Rcpp::Rcout<<"warning: NaN or non-finite eta"<<std::endl;
	break;
      }
      if (!exprm.valideta(eta[i])){
	Rcpp::Rcout<<"warning: eta is not in the support"<<std::endl;
	break;
      }
  }

  //check the validity of mu for Poisson and Binomial family
  double eps = 10. * datum::eps;
  if(exprm.model_name=="poisson")
    if (any(vectorise(mu) < eps))
      Rcpp::Rcout<<"warning: implicit.fit: fitted rates numerically 0 occurred"<<std::endl;
  if(exprm.model_name=="binomial")
      if (any(vectorise(mu) < eps) or any(vectorise(mu) > (1-eps)))
        Rcpp::Rcout<<"warning: implicit.fit: fitted rates numerically 0 occurred"<<std::endl;

  //calculate the deviance
  double dev = exprm.deviance(data.Y, mu, exprm.weights);

  //check the convergence of the algorithm
  bool converged = true;
  if (exprm.convergence){
    mat old_eta;
    mat old_mu;
    old_eta = data.X * out.estimates.col(out.estimates.n_cols-2);
    old_mu = exprm.h_transfer(old_eta);
    double dev2 = exprm.deviance(data.Y, old_mu, exprm.weights);
    if (std::abs(dev-dev2) > exprm.epsilon){
      Rcpp::Rcout<<"warning: implicit.fit: algorithm did not converge"<<std::endl;
      converged = false;
    }
  }


  mat coef = out.last_estimate();
  //check the number of covariates
  if (X_rank < Imp_dataset_size(data).p)
    coef.rows(X_rank, coef.n_rows-1) = datum::nan;

  return Rcpp::List::create(Rcpp::Named("estimates") = out.estimates,
            Rcpp::Named("last") = out.last_estimate(),
	    Rcpp::Named("mu") = mu, Rcpp::Named("eta") = eta,
	    Rcpp::Named("coefficients") = coef, Rcpp::Named("rank") = X_rank,
	    Rcpp::Named("deviance") = dev, Rcpp::Named("converged") = converged);
  return Rcpp::List();
}
