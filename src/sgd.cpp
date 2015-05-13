// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include "basedef.h"
#include "data.h"
#include "experiment.h"
#include "family.h"
#include "learningrate.h"
#include "transfer.h"
#include <stdlib.h>

// Auxiliary function
template<typename EXPERIMENT>
Rcpp::List run_experiment(SEXP dataset, SEXP algorithm, SEXP verbose, EXPERIMENT exprm, Rcpp::List Experiment);

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
// This file will be compiled with C++11
// BH provides methods to use boost library
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(BH)]]

//return the nsamples and p of a dataset
Sgd_Size Sgd_dataset_size(const Sgd_Dataset& dataset) {
  Sgd_Size size;
  size.nsamples = dataset.X.n_rows;
  size.p = dataset.X.n_cols;
  return size;
}

// return the @t th estimated parameter in @online_out
// Here, t=1 is the first estimate, which in matrix will be its 0-th col
mat Sgd_onlineOutput_estimate(const Sgd_OnlineOutput& online_out, unsigned t) {
  if (t==0) {
    return online_out.initial;
  }
  t = t-1;
  mat column = mat(online_out.estimates.col(t));
  return column;
}

// return the @t th data point in @dataset
Sgd_DataPoint Sgd_get_dataset_point(const Sgd_Dataset& dataset, unsigned t) {
  t = t - 1;
  mat xt = mat(dataset.X.row(t));
  double yt = dataset.Y(t);
  return Sgd_DataPoint(xt, yt);
}

// return the new estimate of parameters, using SGD
template<typename EXPERIMENT>
mat Sgd_sgd_online_algorithm(unsigned t, Sgd_OnlineOutput& online_out,
	const Sgd_Dataset& data_history, const EXPERIMENT& experiment, bool& good_gradient) {

  Sgd_DataPoint datapoint = Sgd_get_dataset_point(data_history, t);
  mat theta_old = Sgd_onlineOutput_estimate(online_out, t-1);
  Sgd_Learn_Rate_Value at = experiment.learning_rate(theta_old, datapoint, experiment.offset[t-1], t);
  mat score_t = experiment.score_function(theta_old, datapoint, experiment.offset[t-1]);
  if (!is_finite(score_t))
    good_gradient = false;
#if DEBUG
  static int count = 0;
  if (count < 10) {
    Rcpp::Rcout << "learning rate: \n" << at << std::endl;
    Rcpp::Rcout << "Score function: \n" << score_t << std::endl;
  }
    ++count;
#endif
  mat theta_new = theta_old + mat(at * score_t);
  online_out.estimates.col(t-1) = theta_new;
  return theta_new;
}

// return the new estimate of parameters, using implicit SGD
// TODO add per model
mat Sgd_implicit_online_algorithm(unsigned t, Sgd_OnlineOutput& online_out,
	const Sgd_Dataset& data_history, const Sgd_Experiment_Glm& experiment) {
  Sgd_DataPoint datapoint= Sgd_get_dataset_point(data_history, t);
  mat theta_old = Sgd_onlineOutput_estimate(online_out, t-1);

  mat theta_new;
  Sgd_Learn_Rate_Value at = experiment.learning_rate(theta_old, datapoint, experiment.offset[t-1], t);
  double average_lr = 0;
  if (at.type == 0) average_lr = at.lr_scalar;
  else{
    vec diag_lr = at.lr_mat.diag();
    for (unsigned i = 0; i < diag_lr.n_elem; ++i) {
      average_lr += diag_lr[i];
    }
    average_lr /= diag_lr.n_elem;
  }

  double normx = dot(datapoint.x, datapoint.x);

  Get_score_coeff<Sgd_Experiment_Glm> get_score_coeff(experiment, datapoint, theta_old, normx, experiment.offset[t-1]);
  Implicit_fn<Sgd_Experiment_Glm> implicit_fn(average_lr, get_score_coeff);

  double rt = average_lr * get_score_coeff(0);
  double lower = 0;
  double upper = 0;
  if (rt < 0) {
      upper = 0;
      lower = rt;
  }
  else{
    double u = 0;
    u = (experiment.g_link(datapoint.y) - dot(theta_old,datapoint.x))/normx;
    upper = std::min(rt, u);
    lower = 0;
  }
  double result;
  if (lower != upper) {
      result = boost::math::tools::schroeder_iterate(implicit_fn, (lower + upper)/2, lower, upper, 14);
  }
  else
    result = lower;
  theta_new = theta_old + result * datapoint.x.t();
  online_out.estimates.col(t-1) = theta_new;

  return theta_new;
}

// transform the output of averaged SGD
void asgd_transform_output(Sgd_OnlineOutput& sgd_onlineOutput) {
	mat avg_estimates(sgd_onlineOutput.estimates.n_rows, 1);
	avg_estimates = Sgd_onlineOutput_estimate(sgd_onlineOutput, 1);
	for (unsigned t = 1; t < sgd_onlineOutput.estimates.n_cols; ++t) {
		avg_estimates = (1. - 1./(double)t) * avg_estimates
						+ 1./((double)t) * Sgd_onlineOutput_estimate(sgd_onlineOutput, t+1);
		// t+1-th data has been averaged in @sgd_onlineOutput.estimate,
		// hence can be used to store instantly
		sgd_onlineOutput.estimates.col(t) = avg_estimates;
	}
}

// TODO add per model
bool validity_check(const Sgd_Dataset& data, const mat& theta, unsigned t, const Sgd_Experiment_Glm& exprm) {
  //check if all estimates are finite
  if (!is_finite(theta)) {
    Rcpp::Rcout<<"warning: non-finite coefficients at iteration "<<t<<std::endl;
  }

  //check if eta is in the support
  double eta = exprm.offset[t-1] + dot(Sgd_get_dataset_point(data, t).x, theta);
  if (!exprm.valideta(eta)) {
    Rcpp::Rcout<<"no valid set of coefficients has been found: please supply starting values"<<t<<std::endl;
    return false;
  }

  //check the variance of the expectation of Y
  double mu_var = exprm.variance(exprm.h_transfer(eta));
  if (!is_finite(mu_var)) {
    Rcpp::Rcout<<"NA in V(mu) in iteration "<<t<<std::endl;
    Rcpp::Rcout<<"current theta: "<<theta<<std::endl;
    Rcpp::Rcout<<"current eta: "<<eta<<std::endl;
    return false;
  }
  if (mu_var == 0) {
    Rcpp::Rcout<<"0 in V(mu) in iteration"<<t<<std::endl;
    Rcpp::Rcout<<"current theta: "<<theta<<std::endl;
    Rcpp::Rcout<<"current eta: "<<eta<<std::endl;
    return false;
  }
  double deviance = 0;
  mat mu;
  mat eta_mat;
  //check the deviance
  if (exprm.dev) {
    eta_mat = data.X * theta + exprm.offset;
    mu = exprm.h_transfer(eta_mat);
    deviance = exprm.deviance(data.Y, mu, exprm.weights);
    if(!is_finite(deviance)) {
      Rcpp::Rcout<<"Deviance is non-finite"<<std::endl;
      return false;
    }
  }
  //print if trace
  if(exprm.trace) {
    if (!exprm.dev) {
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
	SEXP verbose) {
  Rcpp::List Experiment(experiment);

  std::string model_name = Rcpp::as<std::string>(Experiment["name"]);
  Rcpp::List model_attrs = Experiment["model.attrs"];

  if (model_name == "gaussian" || model_name == "poisson" || model_name == "binomial" || model_name == "gamma") {
    Sgd_Experiment_Glm exprm(model_name, model_attrs);
    return run_experiment(dataset, algorithm, verbose, exprm, Experiment);
  } else if (model_name == "...") {
    //Sgd_Experiment_Svm exprm(model_name, model_attrs);
    //return run_experiment(dataset, algorithm, verbose, exprm, Experiment);
    return Rcpp::List();
  } else {
    return Rcpp::List();
  }
}

template<typename EXPERIMENT>
Rcpp::List run_experiment(SEXP dataset, SEXP algorithm, SEXP verbose, EXPERIMENT exprm, Rcpp::List Experiment) {
  Rcpp::List Dataset(dataset);

  Sgd_Dataset data;
  data.X = Rcpp::as<mat>(Dataset["X"]);
  data.Y = Rcpp::as<mat>(Dataset["Y"]);
  std::string algo;
  algo =  Rcpp::as<std::string>(algorithm);

  exprm.n_iters = Rcpp::as<unsigned>(Experiment["niters"]);
  exprm.d = Rcpp::as<unsigned>(Experiment["d"]);
  exprm.offset = Rcpp::as<mat>(Experiment["offset"]);
  exprm.weights = Rcpp::as<mat>(Experiment["weights"]);
  exprm.start = Rcpp::as<mat>(Experiment["start"]);
  exprm.convergence = Rcpp::as<bool>(Experiment["convergence"]);
  exprm.dev = Rcpp::as<bool>(Experiment["deviance"]);
  exprm.trace = Rcpp::as<bool>(Experiment["trace"]);
  exprm.epsilon = Rcpp::as<double>(Experiment["epsilon"]);
  std::string lr = Rcpp::as<std::string>(Experiment["lr"]);
  if (lr == "one-dim") {
    // use the min eigenvalue of the covariance of data as alpha in LR
    // TODO this can be arbitrarily small
    cx_vec eigval;
    cx_mat eigvec;
    // eig_gen(eigval, eigvec, data.covariance());
    // double lr_alpha = min(eigval).real();
    // if (lr_alpha < 1e-8) {
      // lr_alpha = 1; // temp hack
    // }
    double lr_alpha = 1;
    double c;
    if (algo == "asgd" || algo == "ai-sgd") {
      c = 2./3.;
    }
    else {
      c = 1.;
    }
    exprm.init_one_dim_learning_rate(1., lr_alpha, c, 1.);
  }
  else if (lr == "one-dim-eigen") {
    exprm.init_one_dim_eigen_learning_rate();
  }
  else if (lr == "d-dim") {
    exprm.init_ddim_learning_rate(0., 1.);
  }
  else if (lr == "adagrad") {
    exprm.init_ddim_learning_rate(1., .5);
  }

  Sgd_OnlineOutput out(data, exprm.start);
  unsigned nsamples = Sgd_dataset_size(data).nsamples;

  //check if the number of observations is greater than the rank of X
  // unsigned X_rank = rank(data.X);
  // if (X_rank > nsamples) {
  //   Rcpp::Rcout<<"X matrix has rank "<<X_rank<<", but only "
  //       <<nsamples<<" observation"<<std::endl;
  //   return Rcpp::List();
  // }
  unsigned X_rank = nsamples;

  // print out info
  //Rcpp::Rcout << data;
  //Rcpp::Rcout << exprm;
  //Rcpp::Rcout << "    Method: " << algo << std::endl;

  bool good_gradient = true;
  bool good_validity = true;
  for(int t=1; t<=nsamples; ++t) {
    if (algo == "sgd" || algo == "asgd") {
      mat theta = Sgd_sgd_online_algorithm(t, out, data, exprm, good_gradient);
      if (!good_gradient) {
        Rcpp::Rcout<<"NA or infinite gradient"<<std::endl;
        return Rcpp::List();
      }
      good_validity = validity_check(data,theta, t, exprm);
      if (!good_validity)
        return Rcpp::List();
    }
    else if (algo == "implicit" || algo == "ai-sgd") {
      mat theta = Sgd_implicit_online_algorithm(t, out, data, exprm);
      if (!is_finite(theta)) {
        Rcpp::Rcout<<"warning: non-finite coefficients at iteration "<<t<<std::endl;
      }
      good_validity = validity_check(data,theta, t, exprm);
      if (!good_validity)
        return Rcpp::List();
    }
  }
  if (algo == "asgd" || algo == "ai-sgd") {
    asgd_transform_output(out);
  }

  //check the validity of eta for all observations
  mat eta;
  eta = data.X * out.last_estimate() + exprm.offset;
  mat mu;
  mu = exprm.h_transfer(eta);
  for(int i=0; i<eta.n_rows; ++i) {
      if (!is_finite(eta[i])) {
        Rcpp::Rcout<<"warning: NaN or non-finite eta"<<std::endl;
        break;
      }
      if (!exprm.valideta(eta[i])) {
        Rcpp::Rcout<<"warning: eta is not in the support"<<std::endl;
        break;
      }
  }

  //check the validity of mu for Poisson and Binomial family
  double eps = 10. * datum::eps;
  if(exprm.model_name=="poisson")
    if (any(vectorise(mu) < eps))
      Rcpp::Rcout<<"warning: sgd.fit: fitted rates numerically 0 occurred"<<std::endl;
  if(exprm.model_name=="binomial")
      if (any(vectorise(mu) < eps) or any(vectorise(mu) > (1-eps)))
        Rcpp::Rcout<<"warning: sgd.fit: fitted rates numerically 0 occurred"<<std::endl;

  //calculate the deviance
  double dev = exprm.deviance(data.Y, mu, exprm.weights);

  //check the convergence of the algorithm
  bool converged = true;
  if (exprm.convergence) {
    mat old_eta;
    mat old_mu;
    old_eta = data.X * out.estimates.col(out.estimates.n_cols-2);
    old_mu = exprm.h_transfer(old_eta);
    double dev2 = exprm.deviance(data.Y, old_mu, exprm.weights);
    if (std::abs(dev-dev2) > exprm.epsilon) {
      Rcpp::Rcout<<"warning: sgd.fit: algorithm did not converge"<<std::endl;
      converged = false;
    }
  }

  mat coef = out.last_estimate();
  //check the number of covariates
  if (X_rank < Sgd_dataset_size(data).p) {
    for (int i = X_rank; i < coef.n_rows; i++) {
      coef.row(i) = datum::nan;
    }
    //coef.rows(X_rank, coef.n_rows-1) = datum::nan;
  }

  return Rcpp::List::create(
            Rcpp::Named("last") = out.last_estimate(),
	    Rcpp::Named("mu") = mu, Rcpp::Named("eta") = eta,
	    Rcpp::Named("coefficients") = coef, Rcpp::Named("rank") = X_rank,
	    Rcpp::Named("deviance") = dev, Rcpp::Named("converged") = converged);
  return Rcpp::List();
}
