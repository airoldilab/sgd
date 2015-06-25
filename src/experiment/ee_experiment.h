#ifndef EXPERIMENT_EE_EXPERIMENT_H
#define EXPERIMENT_EE_EXPERIMENT_H

#include "basedef.h"
#include "data/data_point.h"
#include "experiment/base_experiment.h"
#include "learn-rate/onedim_learn_rate.h"
#include "learn-rate/onedim_eigen_learn_rate.h"
#include "learn-rate/ddim_learn_rate.h"
#include <boost/shared_ptr.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/bind/bind.hpp>
#include <boost/ref.hpp>
#include <iostream>

using namespace arma;

class ee_experiment : public base_experiment {
  /* Experiment class for estimating equations */
public:
  ee_experiment(std::string m_name, Rcpp::List mp_attrs, Rcpp::Function gr)
  : gr_(gr), base_experiment(m_name, mp_attrs) {
    // TODO
    // if model_attrs["wmatrix"] == NULL {
      int k = 5;
      wmatrix_ = eye<mat>(k, k);
    // } else {
    // wmatrix_ = model_attrs["wmatrix"];
    // }
  }

  // Gradient
  mat gradient(const mat& theta_old, const data_point& data_pt,
    double offset) const {
    Rcpp::NumericVector r_theta_old =
      Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(theta_old));
    Rcpp::NumericVector r_data_pt =
      Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(data_pt.x)); // TODO include both x and y (?)
    Rcpp::NumericMatrix r_out = gr_(r_theta_old, r_data_pt);
    mat out = Rcpp::as<mat>(r_out);
    // TODO include weighting matrix
    return out;
  }

private:
  grad_func_type create_grad_func_instance() {
    grad_func_type grad_func = boost::bind(&ee_experiment::gradient, this, _1, _2, _3);
    return grad_func;
  }

  mat wmatrix_;
  Rcpp::Function gr_;
  //TODO look into how optim calls its C function, maybe it stores it too
};

#endif
