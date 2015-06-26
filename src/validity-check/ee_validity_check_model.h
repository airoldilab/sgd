#ifndef VALIDITY_CHECK_EE_VALIDITY_CHECK_MODEL_H
#define VALIDITY_CHECK_EE_VALIDITY_CHECK_MODEL_H

#include "basedef.h"
#include "data/data_set.h"
#include "experiment/ee_experiment.h"

bool validity_check_model(const data_set& data, const mat& theta, unsigned t,
  const ee_experiment& exprm) {
  // TODO
  Rcpp::Rcout << "validity check for EE not implemented" << std::endl;
  return true;
}

#endif
