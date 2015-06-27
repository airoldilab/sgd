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

typedef boost::function<mat(const mat&, const data_point&, double)> grad_func_type;

class ee_experiment : public base_experiment {
  /**
   * Estimating equations
   */
public:
  ee_experiment(std::string m_name, Rcpp::List mp_attrs) :
    base_experiment(m_name, mp_attrs) {
    //gr_(mp_attrs["gr"]), base_experiment(m_name, mp_attrs) {
      //gr_ = model_attrs["gr"];
    // TODO
    // if model_attrs["wmatrix"] == NULL {
      int k = 5;
      wmatrix_ = eye<mat>(k, k);
    // } else {
    // wmatrix_ = model_attrs["wmatrix"];
    // }
  }

  // Gradient
  #if 0
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

  grad_func_type grad_func();
  #endif

  // TODO
  bool rank;
private:
  mat wmatrix_;
  //Rcpp::Function gr_;
};

#endif
