#ifndef POST_PROCESS_EE_POST_PROCESS_H
#define POST_PROCESS_EE_POST_PROCESS_H

#include "basedef.h"
#include "data/data_set.h"
#include "model/ee_model.h"

// model.out: flag to include weighting matrix
template <typename SGD>
Rcpp::List post_process(const SGD& sgd, const data_set& data,
  const ee_model& model) {
  // TODO
  return Rcpp::List();
}

#endif
