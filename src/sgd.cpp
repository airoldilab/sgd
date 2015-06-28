#include "basedef.h"
#include "algorithm/explicit_sgd.h"
#include "algorithm/implicit_sgd.h"
#include "data/data_point.h"
#include "data/data_set.h"
#include "data/online_output.h"
#include "experiment/ee_experiment.h"
#include "experiment/glm_experiment.h"
#include "learn-rate/onedim_learn_rate.h"
#include "learn-rate/onedim_eigen_learn_rate.h"
#include "learn-rate/ddim_learn_rate.h"
#include "post-process/glm_post_process.h"
#include "post-process/ee_post_process.h"
#include "validity-check/validity_check.h"
#include <stdlib.h>

// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

template<typename EXPERIMENT>
void set_experiment(EXPERIMENT& exprm, const Rcpp::List& Experiment);
template<typename EXPERIMENT>
Rcpp::List run_experiment(data_set& data, EXPERIMENT& exprm, std::string method,
  bool verbose, const boost::timer& ti);

// [[Rcpp::export]]
Rcpp::List run(SEXP dataset, SEXP experiment, SEXP method, SEXP verbose) {
  /**
   * Runs the proposed experiment and method on the data set.
   * This is the main interfacing function in R.
   */
  boost::timer ti;
  // Convert all arguments from R to C++ types.
  Rcpp::List Experiment(experiment);
  std::string model_name = Rcpp::as<std::string>(Experiment["name"]);
  Rcpp::List model_attrs = Experiment["model.attrs"];

  Rcpp::List Data(dataset);
  data_set data(Data["bigmat"],
                Rcpp::as<bool>(Data["big"]),
                Rcpp::as<mat>(Data["X"]),
                Rcpp::as<mat>(Data["Y"]),
                Rcpp::as<unsigned>(Experiment["npasses"]));

  std::string meth = Rcpp::as<std::string>(method);
  bool verb = Rcpp::as<bool>(verbose);

  if (model_name == "gaussian" ||
      model_name == "poisson" ||
      model_name == "binomial" ||
      model_name == "gamma") {
    glm_experiment exprm(model_name, model_attrs);
    set_experiment(exprm, Experiment);
    return run_experiment(data, exprm, meth, verb, ti);
  } else if (model_name == "ee") {
    ee_experiment exprm(model_name, model_attrs, model_attrs["gr"]);
    //ee_experiment exprm(model_name, model_attrs);
    set_experiment(exprm, Experiment);
    return run_experiment(data, exprm, meth, verb, ti);
  } else {
    return Rcpp::List();
  }
}

template<typename EXPERIMENT>
void set_experiment(EXPERIMENT& exprm, const Rcpp::List& Experiment) {
  /* Put attributes into experiment. */
  exprm.n_iters = Rcpp::as<unsigned>(Experiment["niters"]);
  exprm.d = Rcpp::as<unsigned>(Experiment["d"]);
  exprm.n_passes = Rcpp::as<unsigned>(Experiment["npasses"]);
  exprm.lr = Rcpp::as<std::string>(Experiment["lr"]);
  exprm.start = Rcpp::as<mat>(Experiment["start"]);
  exprm.weights = Rcpp::as<mat>(Experiment["weights"]);
  exprm.delta = Rcpp::as<double>(Experiment["delta"]);
  exprm.lambda1 = Rcpp::as<double>(Experiment["lambda1"]);
  exprm.lambda2 = Rcpp::as<double>(Experiment["lambda2"]);
  exprm.trace = Rcpp::as<bool>(Experiment["trace"]);
  exprm.dev = Rcpp::as<bool>(Experiment["deviance"]);
  exprm.convergence = Rcpp::as<bool>(Experiment["convergence"]);

  // Set learning rate in experiment.
  vec lr_control= Rcpp::as<vec>(Experiment["lr.control"]);
  if (exprm.lr == "one-dim") {
    exprm.set_learn_rate(new
      onedim_learn_rate(lr_control(0), lr_control(1),
                        lr_control(2), lr_control(3))
      );
  } else if (exprm.lr == "one-dim-eigen") {
    exprm.set_learn_rate(new
      onedim_eigen_learn_rate(exprm.d, exprm.grad_func())
      );
  } else if (exprm.lr == "d-dim") {
    exprm.set_learn_rate(new
      ddim_learn_rate(exprm.d, 1., 0., 1., 1.,
                      lr_control(0), exprm.grad_func())
      );
  } else if (exprm.lr == "adagrad") {
    exprm.set_learn_rate(new
      ddim_learn_rate(exprm.d, lr_control(0), 1., 1., .5,
                      lr_control(1), exprm.grad_func())
      );
  } else if (exprm.lr == "rmsprop") {
    exprm.set_learn_rate(new
      ddim_learn_rate(exprm.d, lr_control(0), lr_control(1),
                      1-lr_control(1), .5, lr_control(2),
                      exprm.grad_func())
      );
  }
}

template<typename EXPERIMENT>
Rcpp::List run_experiment(data_set& data, EXPERIMENT& exprm, std::string method,
  bool verbose, const boost::timer& ti) {
  unsigned n_samples = data.n_samples;
  unsigned n_features = data.n_features;
  unsigned n_passes = exprm.n_passes;

  unsigned X_rank = n_features;
  if (exprm.model_name == "gaussian" ||
      exprm.model_name == "poisson" ||
      exprm.model_name == "binomial" ||
      exprm.model_name == "gamma") {
    // Check if the number of observations is greater than the rank of X.
    if (exprm.rank) {
      X_rank = arma::rank(data.X);
      if (X_rank > n_samples) {
        Rcpp::Rcout << "X matrix has rank " << X_rank << ", but only "
          << n_samples << " observation" << std::endl;
        return Rcpp::List();
      }
    }
  }

  // Initialize booleans.
  bool good_gradient = true;
  bool good_validity = true;
  bool flag_ave = false;
  if (method == "asgd" || method == "ai-sgd") {
    flag_ave = true;
  }

  // Initialize estimates.
  online_output out(data, exprm.start, ti, n_passes);
  mat theta_new;
  mat theta_old = out.get_last_estimate();
  mat theta_new_ave;
  mat theta_old_ave;

  // Run SGD!
  if (verbose) {
    Rcpp::Rcout << "Stochastic gradient method: " << method << std::endl;
    Rcpp::Rcout << "SGD Start! " <<std::endl;
  }
  for (int t = 1; t <= n_samples*n_passes; ++t) {
    // SGD update
    if (method == "sgd" || method == "asgd") {
      theta_new = explicit_sgd(t, theta_old, data, exprm, good_gradient);
    } else if (method == "implicit" || method == "ai-sgd") {
      theta_new = implicit_sgd(t, theta_old, data, exprm, good_gradient);
    }

    // Whether to do averaging
    if (flag_ave) {
      if (t != 1) {
        theta_new_ave = (1. - 1./(double)t) * theta_old_ave +
          1./((double)t) * theta_new;
      } else {
        theta_new_ave = theta_new;
      }
      out = theta_new_ave;
      theta_old_ave = theta_new_ave;
    } else {
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
  Rcpp::List model_out = post_process(out, data, exprm, coef, X_rank);

  return Rcpp::List::create(
    Rcpp::Named("coefficients") = coef,
    Rcpp::Named("converged") = true,
    Rcpp::Named("estimates") = out.get_estimates(),
    Rcpp::Named("times") = out.get_times(),
    Rcpp::Named("pos") = out.get_pos(),
    Rcpp::Named("model.out") = model_out);
}
