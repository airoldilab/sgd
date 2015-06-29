#ifndef POST_PROCESS_EE_POST_PROCESS_H
#define POST_PROCESS_EE_POST_PROCESS_H

#include "basedef.h"
#include "data/data_set.h"
#include "model/ee_model.h"
#include "sgd/sgd.h"
#include <stdlib.h>

// model.out: flag to include weighting matrix
Rcpp::List post_process(const sgd& sgd_out, const data_set& data,
  const ee_model& model, mat& coef, unsigned X_rank) {
  // TODO
  Rcpp::Rcout << "warning: post_process for EE not implemented yet" << std::endl;
  return Rcpp::List::create();
}

#endif
