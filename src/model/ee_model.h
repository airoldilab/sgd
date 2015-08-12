#ifndef MODEL_EE_MODEL_H
#define MODEL_EE_MODEL_H

#include "basedef.h"
#include "data/data_point.h"
#include "model/base_model.h"
#include <boost/shared_ptr.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/ref.hpp>
#include <iostream>

class ee_model : public base_model {
  /**
   * Estimating equations
   *
   * @param model attributes affiliated with model as R type
   */
public:
  ee_model(Rcpp::List model) :
    base_model(model), gr_(Rcpp::as<Rcpp::Function>(model["gr"])) {
    //if model["wmatrix"] == NULL {
      int k = 5;
      wmatrix_ = eye<mat>(k, k);
    //} else {
    //  wmatrix_ = Rcpp::as<mat>(model["wmatrix"]);
    //}
  }

  mat gradient(unsigned t, const mat& theta_old, const data_set& data)
    const {
    data_point data_pt = data.get_data_point(t);
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

  // TODO
  bool rank;

private:
  mat wmatrix_;
  Rcpp::Function gr_;
};

#endif
