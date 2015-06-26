#ifndef POST_PROCESS_EE_POST_PROCESS_H
#define POST_PROCESS_EE_POST_PROCESS_H

#include "basedef.h"
#include "data/data_set.h"
#include "data/online_output.h"
#include "experiment/ee_experiment.h"
#include <stdlib.h>

// model.out: flag to include weighting matrix
Rcpp::List post_process(const online_output& out, const data_set& data,
  const ee_experiment& exprm, mat& coef, unsigned X_rank) {
  // TODO
  Rcpp::Rcout << "post_process for EE not implemented yet" << std::endl;
  return Rcpp::List::create();
}

#endif
