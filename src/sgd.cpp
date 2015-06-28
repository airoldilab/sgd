#include "basedef.h"
#include "algorithm/explicit_sgd.h"
#include "algorithm/implicit_sgd.h"
#include "data/data_point.h"
#include "data/data_set.h"
#include "data/online_output.h"
#include "experiment/ee_experiment.h"
#include "experiment/glm_experiment.h"
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

/**
 * Runs the proposed experiment (model) and method on the data set
 *
 * @param dataset    data set
 * @param experiment list of attributes about model
 * @param method     stochastic gradient method
 * @param verbose    whether to print progress
 */
// [[Rcpp::export]]
Rcpp::List run(SEXP dataset, SEXP experiment, SEXP method, SEXP verbose) {
  boost::timer ti;
  // Convert all arguments from R to C++ types.
  Rcpp::List Experiment(experiment);
  Rcpp::List Data(dataset);
  data_set data(Data["bigmat"],
                Rcpp::as<bool>(Data["big"]),
                Rcpp::as<mat>(Data["X"]),
                Rcpp::as<mat>(Data["Y"]),
                Rcpp::as<unsigned>(Experiment["npasses"]));
  std::string meth = Rcpp::as<std::string>(method);
  bool verb = Rcpp::as<bool>(verbose);

  // Run templated experiment based on the model.
  std::string model_name = Rcpp::as<std::string>(Experiment["name"]);
  if (model_name == "gaussian" ||
      model_name == "poisson" ||
      model_name == "binomial" ||
      model_name == "gamma") {
    glm_experiment exprm(Experiment);
    return run_experiment(data, exprm, meth, verb, ti);
  } else if (model_name == "ee") {
    Rcpp::List model_attrs = Experiment["model.attrs"];
    Rcpp::Function gr = model_attrs["gr"];
    ee_experiment exprm(Experiment, gr);
    return run_experiment(data, exprm, meth, verb, ti);
  } else {
    return Rcpp::List();
  }
}

/**
 * Runs algorithm templated on the particular experiment type
 *
 * @param  data       data set
 * @tparam EXPERIMENT list of attributes about model
 * @param  method     stochastic gradient method
 * @param  verbose    whether to print progress
 * @param  ti         timer to benchmark progress
 */
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
