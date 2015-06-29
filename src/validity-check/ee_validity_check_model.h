#ifndef VALIDITY_CHECK_EE_VALIDITY_CHECK_MODEL_H
#define VALIDITY_CHECK_EE_VALIDITY_CHECK_MODEL_H

#include "basedef.h"
#include "data/data_set.h"
#include "model/ee_model.h"

bool validity_check_model(const data_set& data, const mat& theta, unsigned t,
  const ee_model& model) {
  // TODO
  Rcpp::Rcout << "warning: validity check for EE not implemented yet" << std::endl;
  return true;
}

#endif
