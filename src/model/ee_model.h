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
  // TODO interface not consistent with other models
  // Constructors
  ee_model(Rcpp::List model, Rcpp::Function gr) :
    base_model(model), gr_(gr) {
    // if model_attrs["wmatrix"] == NULL {
      int k = 5;
      wmatrix_ = eye<mat>(k, k);
    // } else {
    // wmatrix_ = model_attrs["wmatrix"];
    // }
  }

  // Gradient
  mat gradient(const mat& theta_old, const data_point& data_pt) const {
    Rcpp::NumericVector r_theta_old =
      Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(theta_old));
    Rcpp::NumericVector r_data_pt =
      Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(data_pt.x)); // TODO include both x and y (?)
    Rcpp::NumericMatrix r_out = gr_(r_theta_old, r_data_pt);
    mat out = Rcpp::as<mat>(r_out);
    // TODO include weighting matrix
    return out;
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
