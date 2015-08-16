#include "basedef.h"
#include "data/data_set.h"
#include "model/cox_model.h"
#include "model/ee_model.h"
#include "model/glm_model.h"
#include "post-process/cox_post_process.h"
#include "post-process/ee_post_process.h"
#include "post-process/glm_post_process.h"
#include "sgd/explicit_sgd.h"
#include "sgd/implicit_sgd.h"
#include "sgd/momentum_sgd.h"
#include "sgd/nesterov_sgd.h"
#include "validity-check/validity_check.h"
#include <stdlib.h>

// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

template<typename MODEL, typename SGD>
Rcpp::List run(const data_set& data, MODEL& model, SGD& sgd);

/**
 * Runs the proposed model and stochastic gradient method on the data set
 *
 * @param dataset       data set
 * @param model_control attributes affiliated with model
 * @param sgd_control   attributes affiliated with sgd
 */
// [[Rcpp::export]]
Rcpp::List run(SEXP dataset, SEXP model_control, SEXP sgd_control) {
  boost::timer ti;
  Rcpp::List Dataset(dataset);
  Rcpp::List Model_control(model_control);
  Rcpp::List Sgd_control(sgd_control);
  if (Rcpp::as<bool>(Sgd_control["verbose"])) {
    Rcpp::Rcout << "Converting arguments from R to C++ types..." << std::endl;
  }

  // Construct data.
  data_set data(Dataset["bigmat"],
                Rcpp::as<bool>(Dataset["big"]),
                Rcpp::as<mat>(Dataset["X"]),
                Rcpp::as<mat>(Dataset["Y"]),
                Rcpp::as<unsigned>(Sgd_control["npasses"]));

  // Construct model.
  std::string model_name = Rcpp::as<std::string>(Model_control["name"]);
  if (model_name == "cox") {
    cox_model model(Model_control);
    // Construct stochastic gradient method.
    std::string sgd_name = Rcpp::as<std::string>(Sgd_control["method"]);
    if (sgd_name == "sgd" || sgd_name == "asgd") {
      explicit_sgd sgd(Sgd_control, data.n_samples, ti);
      return run(data, model, sgd);
    } else if (sgd_name == "implicit" || sgd_name == "ai-sgd") {
      implicit_sgd sgd(Sgd_control, data.n_samples, ti);
      return run(data, model, sgd);
    } else if (sgd_name == "momentum") {
      momentum_sgd sgd(Sgd_control, data.n_samples, ti);
      return run(data, model, sgd);
    } else if (sgd_name == "nesterov") {
      nesterov_sgd sgd(Sgd_control, data.n_samples, ti);
      return run(data, model, sgd);
    } else {
      Rcpp::Rcout << "error: stochastic gradient method not implemented" << std::endl;
      return Rcpp::List();
    }
  } else if (model_name == "ee") {
    ee_model model(Model_control);
    // Construct stochastic gradient method.
    std::string sgd_name = Rcpp::as<std::string>(Sgd_control["method"]);
    if (sgd_name == "sgd" || sgd_name == "asgd") {
      explicit_sgd sgd(Sgd_control, data.n_samples, ti);
      return run(data, model, sgd);
    } else if (sgd_name == "implicit" || sgd_name == "ai-sgd") {
      implicit_sgd sgd(Sgd_control, data.n_samples, ti);
      return run(data, model, sgd);
    } else if (sgd_name == "momentum") {
      momentum_sgd sgd(Sgd_control, data.n_samples, ti);
      return run(data, model, sgd);
    } else if (sgd_name == "nesterov") {
      nesterov_sgd sgd(Sgd_control, data.n_samples, ti);
      return run(data, model, sgd);
    } else {
      Rcpp::Rcout << "error: stochastic gradient method not implemented" << std::endl;
      return Rcpp::List();
    }
  } else if (model_name == "lm" || model_name == "glm") {
    glm_model model(Model_control);
    // Construct stochastic gradient method.
    std::string sgd_name = Rcpp::as<std::string>(Sgd_control["method"]);
    if (sgd_name == "sgd" || sgd_name == "asgd") {
      explicit_sgd sgd(Sgd_control, data.n_samples, ti);
      return run(data, model, sgd);
    } else if (sgd_name == "implicit" || sgd_name == "ai-sgd") {
      implicit_sgd sgd(Sgd_control, data.n_samples, ti);
      return run(data, model, sgd);
    } else if (sgd_name == "momentum") {
      momentum_sgd sgd(Sgd_control, data.n_samples, ti);
      return run(data, model, sgd);
    } else if (sgd_name == "nesterov") {
      nesterov_sgd sgd(Sgd_control, data.n_samples, ti);
      return run(data, model, sgd);
    } else {
      Rcpp::Rcout << "error: stochastic gradient method not implemented" << std::endl;
      return Rcpp::List();
    }
  } else {
    Rcpp::Rcout << "error: model not implemented" << std::endl;
    return Rcpp::List();
  }

  #if 0
  // TODO The above duplicates code within the if-else statement.
  // Construct stochastic gradient method.
  std::string sgd_name = Rcpp::as<std::string>(Sgd_control["method"]);
  if (sgd_name == "sgd" || sgd_name == "asgd") {
    explicit_sgd sgd(Sgd_control, data.n_samples, ti);
    return run(data, model, sgd);
  } else if (sgd_name == "implicit" || sgd_name == "ai-sgd") {
    implicit_sgd sgd(Sgd_control, data.n_samples, ti);
    return run(data, model, sgd);
  } else if (sgd_name == "momentum") {
    momentum_sgd sgd(Sgd_control, data.n_samples, ti);
    return run(data, model, sgd);
  } else if (sgd_name == "nesterov") {
    nesterov_sgd sgd(Sgd_control, data.n_samples, ti);
    return run(data, model, sgd);
  } else {
    Rcpp::Rcout << "error: stochastic gradient method not implemented" << std::endl;
    return Rcpp::List();
  }
  #endif
}

/**
 * Runs algorithm templated on the model and stochastic gradient method
 *
 * @param  data     data set
 * @tparam MODEL    model class
 * @tparam SGD      stochastic gradient descent class
 */
template<typename MODEL, typename SGD>
Rcpp::List run(const data_set& data, MODEL& model, SGD& sgd) {
  unsigned n_samples = data.n_samples;
  unsigned n_features = data.n_features;
  unsigned n_passes = sgd.get_n_passes();

  // TODO this shouldn't be placed here
  unsigned X_rank = n_features;
  if (model.name() == "lm" || model.name() == "glm") {
    // Check if the number of observations is greater than the rank of X.
    if (model.rank) {
      X_rank = arma::rank(data.X);
      if (X_rank > n_samples) {
        Rcpp::Rcout << "error: X matrix has rank " << X_rank << ", but only "
          << n_samples << " observation" << std::endl;
        return Rcpp::List();
      }
    }
  }

  bool good_gradient = true;
  bool good_validity = true;
  bool flag_ave = false;
  if (sgd.name() == "asgd" || sgd.name() == "ai-sgd") {
    flag_ave = true;
  }

  mat theta_new;
  mat theta_old = sgd.get_last_estimate();
  mat theta_new_ave;
  mat theta_old_ave;

  if (sgd.verbose()) {
    Rcpp::Rcout << "Stochastic gradient method: " << sgd.name() << std::endl;
    Rcpp::Rcout << "SGD Start!" << std::endl;
  }
  for (int t = 1; t <= n_samples*n_passes; ++t) {
    theta_new = sgd.update(t, theta_old, data, model, good_gradient);

    if (flag_ave) {
      if (t != 1) {
        theta_new_ave = (1. - 1./(double)t) * theta_old_ave +
          1./((double)t) * theta_new;
      } else {
        theta_new_ave = theta_new;
      }
      sgd = theta_new_ave;
      theta_old_ave = theta_new_ave;
    } else {
      sgd = theta_new;
    }
    theta_old = theta_new;

    good_validity = validity_check(data, theta_old, good_gradient, t, model);
    if (!good_validity) {
      return Rcpp::List();
    }
  }

  mat coef = sgd.get_last_estimate();
  Rcpp::List model_out = post_process(sgd, data, model, coef, X_rank);

  return Rcpp::List::create(
    Rcpp::Named("model") = model.name(),
    Rcpp::Named("coefficients") = coef,
    Rcpp::Named("converged") = true,
    Rcpp::Named("estimates") = sgd.get_estimates(),
    Rcpp::Named("pos") = sgd.get_pos(),
    Rcpp::Named("times") = sgd.get_times(),
    Rcpp::Named("model.out") = model_out);
}
