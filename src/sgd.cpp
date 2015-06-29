#include "basedef.h"
#include "algorithm/explicit_sgd.h"
#include "algorithm/implicit_sgd.h"
#include "data/data_point.h"
#include "data/data_set.h"
#include "model/ee_model.h"
#include "model/glm_model.h"
#include "post-process/glm_post_process.h"
#include "post-process/ee_post_process.h"
#include "sgd/sgd.h"
#include "validity-check/validity_check.h"
#include <stdlib.h>

// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

template<typename MODEL>
Rcpp::List run_model(const data_set& data, MODEL& model, sgd& sgd_out,
  bool verbose);

/**
 * Runs the proposed model and method on the data set
 *
 * @param dataset    data set
 * @param experiment list of attributes about model/sgd
 * @param method     stochastic gradient method
 * @param verbose    whether to print progress
 */
// TODO split experiment into model and sgd attributes, have constructors wrap
// around them
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
  sgd sgd_out(data.n_samples,
              data.n_features,
              Rcpp::as<unsigned>(Experiment["npasses"]),
              Rcpp::as<mat>(Experiment["start"]),
              Rcpp::as<std::string>(method),
              Rcpp::as<double>(Experiment["delta"]),
              Rcpp::as<bool>(Experiment["convergence"]),
              ti);
  bool verb = Rcpp::as<bool>(verbose);

  // Run templated model.
  std::string name = Rcpp::as<std::string>(Experiment["name"]);
  if (name == "gaussian" ||
      name == "poisson" ||
      name == "binomial" ||
      name == "gamma") {
    glm_model model(Experiment);
    sgd_out.set_learn_rate(Rcpp::as<std::string>(Experiment["lr"]),
                           Rcpp::as<vec>(Experiment["lr.control"]),
                           data.n_features,
                           model.grad_func());
    return run_model(data, model, sgd_out, verb);
  } else if (name == "ee") {
    Rcpp::List model_attrs = Experiment["model.attrs"];
    Rcpp::Function gr = model_attrs["gr"];
    ee_model model(Experiment, gr);
    sgd_out.set_learn_rate(Rcpp::as<std::string>(Experiment["lr"]),
                           Rcpp::as<vec>(Experiment["lr.control"]),
                           data.n_features,
                           model.grad_func());
    return run_model(data, model, sgd_out, verb);
  } else {
    return Rcpp::List();
  }
}

/**
 * Runs algorithm templated on the particular model
 *
 * @param  data     data set
 * @tparam MODEL    model class
 * @param  sgd_out  values and functions affiliated with sgd
 * @param  verbose  whether to print progress
 */
template<typename MODEL>
Rcpp::List run_model(const data_set& data, MODEL& model, sgd& sgd_out,
  bool verbose) {
  unsigned n_samples = data.n_samples;
  unsigned n_features = data.n_features;
  unsigned n_passes = sgd_out.get_n_passes();
  std::string method = sgd_out.get_method();

  unsigned X_rank = n_features;
  if (model.name == "gaussian" ||
      model.name == "poisson" ||
      model.name == "binomial" ||
      model.name == "gamma") {
    // Check if the number of observations is greater than the rank of X.
    if (model.rank) {
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
  mat theta_new;
  mat theta_old = sgd_out.get_last_estimate();
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
      theta_new = explicit_sgd(t, theta_old, data, model, sgd_out, good_gradient);
    } else if (method == "implicit" || method == "ai-sgd") {
      theta_new = implicit_sgd(t, theta_old, data, model, sgd_out, good_gradient);
    }

    // Whether to do averaging
    if (flag_ave) {
      if (t != 1) {
        theta_new_ave = (1. - 1./(double)t) * theta_old_ave +
          1./((double)t) * theta_new;
      } else {
        theta_new_ave = theta_new;
      }
      sgd_out = theta_new_ave;
      theta_old_ave = theta_new_ave;
    } else {
      sgd_out = theta_new;
    }
    theta_old = theta_new;

    // Validity check
    good_validity = validity_check(data, theta_old, good_gradient, t, model);
    if (!good_validity) {
      return Rcpp::List();
    }
  }

  // Collect model-specific output.
  mat coef = sgd_out.get_last_estimate();
  Rcpp::List model_out = post_process(sgd_out, data, model, coef, X_rank);

  return Rcpp::List::create(
    Rcpp::Named("coefficients") = coef,
    Rcpp::Named("converged") = true,
    Rcpp::Named("estimates") = sgd_out.get_estimates(),
    Rcpp::Named("times") = sgd_out.get_times(),
    Rcpp::Named("pos") = sgd_out.get_pos(),
    Rcpp::Named("model.out") = model_out);
}
