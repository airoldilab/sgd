// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include "basedef.h"
#include "data.h"
#include "experiment.h"
#include "glm-family.h"
#include "glm-transfer.h"
#include "learningrate.h"
#include <stdlib.h>

// Auxiliary function
template<typename EXPERIMENT>
Rcpp::List run_experiment(Sgd_Dataset data, EXPERIMENT exprm, std::string method, bool verbose, Rcpp::List Experiment);

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
// This file will be compiled with C++11
// BH provides methods to use boost library
//
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(BH)]]

// return the nsamples and nfeatures of a dataset
Sgd_Size Sgd_dataset_size(const Sgd_Dataset& dataset) {
  Sgd_Size size;
  size.nsamples = dataset.n_samples;
  size.d = dataset.X.n_cols;
  return size;
}

// return the @t th data point in @dataset
Sgd_DataPoint Sgd_get_dataset_point(const Sgd_Dataset& dataset, unsigned t) {
  t = t - 1;
  mat xt = mat(dataset.X.row(dataset.idxmap[t]));
  
  
  double yt = dataset.Y(dataset.idxmap[t]);
  return Sgd_DataPoint(xt, yt);
}

// return the new estimate of parameters, using SGD
template<typename EXPERIMENT>
mat Sgd_sgd_online_algorithm(unsigned t, const mat& theta_old,
  const Sgd_Dataset& data_history, const EXPERIMENT& experiment,
  bool& good_gradient) {

  Sgd_DataPoint datapoint = Sgd_get_dataset_point(data_history, t);
  unsigned idx = data_history.idxmap[t-1];
  Sgd_Learn_Rate_Value at = experiment.learning_rate(theta_old, datapoint, experiment.offset[idx], t);
  mat grad_t = experiment.gradient(theta_old, datapoint, experiment.offset[idx]);
  if (!is_finite(grad_t)) {
    good_gradient = false;
  }
  mat theta_new = theta_old + (at * grad_t);

  // check the correctness of SGD update in DEBUG mode
#if DEBUG
  if (!(at < 1)){
    Rcpp::Rcout << "learning rate: " << at <<
      "at Iter: " << t << std::endl;
  }
  mat theta_test;
  if (experiment.model_name == "gaussian" || experiment.model_name == "poisson"
    || experiment.model_name == "binomial" || experiment.model_name == "gamma"){
    theta_test = theta_old + at * ((datapoint.y - experiment.h_transfer(
      dot(datapoint.x, theta_old) + experiment.offset[idx]))*datapoint.x).t();
  } else{
    theta_test = theta_new;
  }
  double error = max(max(abs(theta_test - theta_new)));
  double scale = max(max(abs(theta_test)));
  if (error/scale > 1e-5) {
    Rcpp::Rcout<< "Wrong SGD update at iter: " << t + 1 << std::endl;
    Rcpp::Rcout<< "Relative Error = " <<  max(max(abs(theta_test - theta_new))) << std::endl;
    Rcpp::Rcout<< "Correct = " << theta_test << std::endl;
    Rcpp::Rcout<< "Output = " << theta_new << std::endl;
  }
#endif
  return theta_new;
}

// return the new estimate of parameters, using implicit SGD
// TODO add per model
mat Sgd_implicit_online_algorithm(unsigned t, const mat& theta_old,
  const Sgd_Dataset& data_history, const Sgd_Experiment_Glm& experiment,
  bool& good_gradient) {
  Sgd_DataPoint datapoint= Sgd_get_dataset_point(data_history, t);
  mat theta_new;
  unsigned idx = data_history.idxmap[t-1];
  Sgd_Learn_Rate_Value at = experiment.learning_rate(theta_old, datapoint, experiment.offset[idx], t);
  double average_lr = 0;
  if (at.type == 0) average_lr = at.lr_scalar;
  else {
    vec diag_lr = at.lr_mat.diag();
    for (unsigned i = 0; i < diag_lr.n_elem; ++i) {
      average_lr += diag_lr[i];
    }
    average_lr /= diag_lr.n_elem;
  }

  double normx = dot(datapoint.x, datapoint.x);

  Get_grad_coeff<Sgd_Experiment_Glm> get_grad_coeff(experiment, datapoint, theta_old, normx, experiment.offset[idx]);
  Implicit_fn<Sgd_Experiment_Glm> implicit_fn(average_lr, get_grad_coeff);

  double rt = average_lr * get_grad_coeff(0);
  double lower = 0;
  double upper = 0;
  if (rt < 0) {
    upper = 0;
    lower = rt;
  }
  else {
    double u = 0;
    u = (experiment.g_link(datapoint.y) - dot(theta_old,datapoint.x))/normx;
    upper = std::min(rt, u);
    lower = 0;
  }
  double result;
  if (lower != upper) {
    result = boost::math::tools::schroeder_iterate(implicit_fn, (lower + upper)/2, lower, upper, experiment.delta);
  }
  else
    result = lower;
  theta_new = theta_old + result * datapoint.x.t();

  // check the correctness of SGD update in DEBUG mode
#if DEBUG
  if (!(at < 1)){
    Rcpp::Rcout << "learning rate: " << at <<
      "at Iter: " << t << std::endl;
  }
  mat theta_test;
  if (experiment.model_name == "gaussian" || experiment.model_name == "poisson"
    || experiment.model_name == "binomial" || experiment.model_name == "gamma"){
    theta_test = theta_new - at * ((datapoint.y - experiment.h_transfer(
      dot(datapoint.x, theta_new) + experiment.offset[idx]))*datapoint.x).t();
  } else{
    theta_test = theta_old;
  }
  double error = max(max(abs(theta_test - theta_old)));
  double scale = max(max(abs(theta_test)));
  if (error/scale > 1e-5) {
    Rcpp::Rcout<< "Wrong SGD update at iter: " << t + 1 << std::endl;
    Rcpp::Rcout<< "Relative Error = " <<  max(max(abs(theta_test - theta_new))) << std::endl;
    Rcpp::Rcout<< "Correct = " << theta_test << std::endl;
    Rcpp::Rcout<< "Output = " << theta_new << std::endl;
  }
#endif
  return theta_new;
}

template<typename EXPERIMENT>
bool validity_check(const Sgd_Dataset& data, const mat& theta,
  bool good_gradient, unsigned t, const EXPERIMENT& exprm) {
  if (!good_gradient) {
    Rcpp::Rcout << "NA or infinite gradient" << std::endl;
    return false;
  }

  // Check if all estimates are finite.
  if (!is_finite(theta)) {
    Rcpp::Rcout << "warning: non-finite coefficients at iteration " << t << std::endl;
  }

  return validity_check_model(data, theta, t, exprm);
}

// TODO add per model
bool validity_check_model(const Sgd_Dataset& data, const mat& theta, unsigned t,
  const Sgd_Experiment_Glm& exprm) {
  // Check if eta is in the support.
  unsigned idx = data.idxmap[t-1];
  double eta = exprm.offset[idx] + dot(Sgd_get_dataset_point(data, t).x, theta);
  if (!exprm.valideta(eta)) {
    Rcpp::Rcout << "no valid set of coefficients has been found: please supply starting values" << t << std::endl;
    return false;
  }

  // Check the variance of the expectation of Y.
  double mu_var = exprm.variance(exprm.h_transfer(eta));
  if (!is_finite(mu_var)) {
    Rcpp::Rcout << "NA in V(mu) in iteration " << t << std::endl;
    Rcpp::Rcout << "current theta: " << theta << std::endl;
    Rcpp::Rcout << "current eta: " << eta << std::endl;
    return false;
  }
  // if (mu_var == 0) {
  //   Rcpp::Rcout << "0 in V(mu) in iteration" << t << std::endl;
  //   Rcpp::Rcout << "current theta: " << theta << std::endl;
  //   Rcpp::Rcout << "current eta: " << eta << std::endl;
  //   return false;
  // }
  double deviance = 0;
  mat mu;
  mat eta_mat;

  // Check the deviance.
  if (exprm.dev) {
    eta_mat = data.X * theta + exprm.offset;
    mu = exprm.h_transfer(eta_mat);
    deviance = exprm.deviance(data.Y, mu, exprm.weights);
    if(!is_finite(deviance)) {
      Rcpp::Rcout << "Deviance is non-finite" << std::endl;
      return false;
    }
  }

  // Print if trace.
  if (exprm.trace) {
    if (!exprm.dev) {
      eta_mat = data.X * theta + exprm.offset;
      mu = exprm.h_transfer(eta_mat);
      deviance = exprm.deviance(data.Y, mu, exprm.weights);
    }
    Rcpp::Rcout << "Deviance = " << deviance << " , Iterations - " << t << std::endl;
  }
  return true;
}

template<typename EXPERIMENT>
Rcpp::List post_process_glm(const Sgd_OnlineOutput& out, const Sgd_Dataset& data,
  const EXPERIMENT& exprm, mat& coef, unsigned X_rank) {
  // Check the validity of eta for all observations.
  mat eta;
  eta = data.X * out.get_last_estimate() + exprm.offset;
  mat mu;
  mu = exprm.h_transfer(eta);
  for (int i = 0; i < eta.n_rows; ++i) {
      if (!is_finite(eta[i])) {
        Rcpp::Rcout << "warning: NaN or non-finite eta" << std::endl;
        break;
      }
      if (!exprm.valideta(eta[i])) {
        Rcpp::Rcout << "warning: eta is not in the support" << std::endl;
        break;
      }
  }

  // Check the validity of mu for Poisson and Binomial family.
  double eps = 10. * datum::eps;
  if (exprm.model_name == "poisson")
    if (any(vectorise(mu) < eps))
      Rcpp::Rcout << "warning: sgd.fit: fitted rates numerically 0 occurred" << std::endl;
  if (exprm.model_name == "binomial")
      if (any(vectorise(mu) < eps) or any(vectorise(mu) > (1-eps)))
        Rcpp::Rcout << "warning: sgd.fit: fitted rates numerically 0 occurred" << std::endl;

  // Calculate the deviance.
  double dev = exprm.deviance(data.Y, mu, exprm.weights);

  // Check the number of features.
  if (X_rank < Sgd_dataset_size(data).d) {
    for (int i = X_rank; i < coef.n_rows; i++) {
      coef.row(i) = datum::nan;
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("mu") = mu,
    Rcpp::Named("eta") = eta,
    Rcpp::Named("rank") = X_rank,
    Rcpp::Named("deviance") = dev);
}

// TODO
// post_process_ee
// model.out: flag to include weighting matrix

// use the method specified by method to estimate parameters
// [[Rcpp::export]]
Rcpp::List run_online_algorithm(SEXP dataset, SEXP experiment, SEXP method,
  SEXP verbose) {
  // Convert all arguments from R to C++ types.
  
  Rcpp::List Experiment(experiment);
  std::string model_name = Rcpp::as<std::string>(Experiment["name"]);
  Rcpp::List model_attrs = Experiment["model.attrs"];

  Rcpp::List Dataset(dataset);
  Sgd_Dataset data;
  data.X = Rcpp::as<mat>(Dataset["X"]);
  data.Y = Rcpp::as<mat>(Dataset["Y"]);
  data.init(Rcpp::as<unsigned>(Experiment["npasses"]));

  std::string meth = Rcpp::as<std::string>(method);
  bool verb = Rcpp::as<bool>(verbose);

  if (model_name == "gaussian" || model_name == "poisson" || model_name == "binomial" || model_name == "gamma") {
    Sgd_Experiment_Glm exprm(model_name, model_attrs);
    return run_experiment(data, exprm, meth, verb, Experiment);
  //} else if (model_name == "ee") {
  //  Sgd_Experiment_Ee exprm(model_name, model_attrs);
  //  return run_experiment(data, exprm, meth, verb, Experiment);
  } else {
    return Rcpp::List();
  }
}

template<typename EXPERIMENT>
Rcpp::List run_experiment(Sgd_Dataset data, EXPERIMENT exprm, std::string method,
  bool verbose, Rcpp::List Experiment) {

  // Put remaining attributes into experiment.
  exprm.n_iters = Rcpp::as<unsigned>(Experiment["niters"]);
  exprm.d = Rcpp::as<unsigned>(Experiment["d"]);
  exprm.n_passes = Rcpp::as<unsigned>(Experiment["npasses"]);
  exprm.lr = Rcpp::as<std::string>(Experiment["lr"]);
  exprm.start = Rcpp::as<mat>(Experiment["start"]);
  exprm.weights = Rcpp::as<mat>(Experiment["weights"]);
  exprm.offset = Rcpp::as<mat>(Experiment["offset"]);
  exprm.delta = Rcpp::as<double>(Experiment["delta"]);
  exprm.trace = Rcpp::as<bool>(Experiment["trace"]);
  exprm.dev = Rcpp::as<bool>(Experiment["deviance"]);
  exprm.convergence = Rcpp::as<bool>(Experiment["convergence"]);


  // Set learning rate in experiment.
  if (exprm.lr == "one-dim") {
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
    if (method == "asgd" || method == "ai-sgd") {
      c = 2./3.;
    }
    else {
      c = 1.;
    }
    exprm.init_one_dim_learning_rate(1., lr_alpha, c, 1.);
  }
  else if (exprm.lr == "one-dim-eigen") {
    exprm.init_one_dim_eigen_learning_rate();
  }
  else if (exprm.lr == "d-dim") {
    exprm.init_ddim_learning_rate(0., 1., 1., 0.000001);
  }
  else if (exprm.lr == "adagrad") {
    exprm.init_ddim_learning_rate(1., 1., .5, 0.000001);
  }
  else if (exprm.lr == "rmsprop") {
    double gamma = 0.9;
    exprm.init_ddim_learning_rate(gamma, 1-gamma, .5, 0.000001);
  }


  unsigned nsamples = Sgd_dataset_size(data).nsamples;
  unsigned ndim = Sgd_dataset_size(data).d;


  // Check if the number of observations is greater than the rank of X.
  unsigned X_rank = ndim;
  if (exprm.model_name == "gaussian" ||
      exprm.model_name == "poisson" ||
      exprm.model_name == "binomial" ||
      exprm.model_name == "gamma") {
    if (exprm.rank) {
      X_rank = rank(data.X);
      if (X_rank > nsamples) {
        Rcpp::Rcout << "X matrix has rank " << X_rank << ", but only "
          << nsamples << " observation" << std::endl;
        return Rcpp::List();
      }
    }
  }

#if DEBUG
  Rcpp::Rcout << data;
  Rcpp::Rcout << exprm;
  Rcpp::Rcout << "    Method: " << method << std::endl;
#endif

  // Initialize booleans.
  bool good_gradient = true;
  bool good_validity = true;
  bool flag_ave;
  if (method == "asgd" || method == "ai-sgd") {
    flag_ave = true;
  }

  // Initialize estimates.
  Sgd_OnlineOutput out(data, exprm.start);
  mat theta_new;
  mat theta_old = out.get_last_estimate();
  mat theta_new_ave;
  mat theta_old_ave;

  // Run SGD!
  #if DEBUG
  Rcpp::Rcout << "SGD Start! " <<std::endl;
  #endif
  for (int t = 1; t <= nsamples; ++t) {
    // SGD update
    if (method == "sgd" || method == "asgd") {
      theta_new = Sgd_sgd_online_algorithm(t, theta_old, data, exprm, good_gradient);
    }
    else if (method == "implicit" || method == "ai-sgd") {
      theta_new = Sgd_implicit_online_algorithm(t, theta_old, data, exprm, good_gradient);
    }

    // Whether to do averaging
    if (flag_ave) {
      if (t != 1) {
        theta_new_ave = (1. - 1./(double)t) * theta_old_ave
          + 1./((double)t) * theta_new;
      } else {
        theta_new_ave = theta_new;
      }
      out = theta_new_ave;
      theta_old_ave = theta_new_ave;
    }
    else {
      out = theta_new;
    }
    theta_old = theta_new;

    // Validity check
    good_validity = validity_check(data, theta_old, good_gradient, t, exprm);
    if (!good_validity) {
      return Rcpp::List();
    }
  }

  // Collect model-specific output.
  mat coef = out.get_last_estimate();
  Rcpp::List model_out;
  if (exprm.model_name == "gaussian" ||
      exprm.model_name == "poisson" ||
      exprm.model_name == "binomial" ||
      exprm.model_name == "gamma") {
    model_out = post_process_glm(out, data, exprm, coef, X_rank);
  }

  return Rcpp::List::create(
    Rcpp::Named("coefficients") = coef,
    Rcpp::Named("converged") = true,
    Rcpp::Named("estimates") = out.estimates,
    Rcpp::Named("pos") = out.pos,
    Rcpp::Named("model.out") = model_out);
}
