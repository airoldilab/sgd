#ifndef POST_PROCESS_GMM_POST_PROCESS_H
#define POST_PROCESS_GMM_POST_PROCESS_H

#include "basedef.h"
#include "data/data_set.h"
#include "model/gmm_model.h"

// model.out: flag to include weighting matrix
template <typename SGD>
Rcpp::List post_process(const SGD& sgd, const data_set& data,
  const gmm_model& model) {
  // TODO
  return Rcpp::List();
}

#endif
