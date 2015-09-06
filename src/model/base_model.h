#ifndef MODEL_BASE_MODEL_H
#define MODEL_BASE_MODEL_H

#include "basedef.h"
#include "data/data_point.h"

class base_model {
  /**
   * Base class for models
   *
   * @param model attributes affiliated with model as R type
   */
public:
  base_model(Rcpp::List model) {
    name_ = Rcpp::as<std::string>(model["name"]);
    lambda1 = Rcpp::as<double>(model["lambda1"]);
    lambda2 = Rcpp::as<double>(model["lambda2"]);
  }

  std::string name() const {
    return name_;
  }

  mat gradient(unsigned t, const mat& theta_old, const data_set& data) const;

  // TODO make private
  double lambda1;
  double lambda2;

protected:
  std::string name_;
};

#endif
