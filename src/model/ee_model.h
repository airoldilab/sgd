#ifndef MODEL_EE_MODEL_H
#define MODEL_EE_MODEL_H

#include "basedef.h"
#include "data/data_point.h"
#include "model/base_model.h"
#include <boost/shared_ptr.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/bind/bind.hpp>
#include <boost/ref.hpp>
#include <iostream>

typedef boost::function<mat(const mat&, const data_point&)> grad_func_type;

class ee_model : public base_model {
  /**
   * Estimating equations
   *
   * @param model attributes affiliated with model as R type
   */
public:
  // Constructors
  ee_model(Rcpp::List model) :
    base_model(model), gr_(Rcpp::as<Rcpp::Function>(model["gr"])) {
    //if model["wmatrix"] == NULL {
      int k = 5;
      wmatrix_ = eye<mat>(k, k);
    //} else {
    //  wmatrix_ = Rcpp::as<mat>(model["wmatrix"]);
    //}
  }

  // Gradient
  mat gradient(const mat& theta_old, const data_point& data_pt) const {
    // TODO y isn't necessary
    // TODO include weighting matrix
    Rcpp::NumericVector r_theta_old =
      Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(theta_old));
    Rcpp::NumericVector r_data_pt =
      Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(data_pt.x));
    Rcpp::NumericMatrix r_out = gr_(r_theta_old, r_data_pt);
    mat out = Rcpp::as<mat>(r_out);
    return -1. * out; // maximize the negative moment function
  }

  // Learning rates
  grad_func_type grad_func() {
    return boost::bind(&ee_model::gradient, this, _1, _2);
  }

  // TODO
  bool rank;

private:
  // Members
  mat wmatrix_;
  Rcpp::Function gr_;
};

#endif
