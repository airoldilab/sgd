#ifndef POST_PROCESS_GLM_POST_PROCESS_H
#define POST_PROCESS_GLM_POST_PROCESS_H

#include "../basedef.h"
#include "../data/data_set.h"
#include "../model/glm_model.h"

template <typename SGD>
Rcpp::List post_process(const SGD& sgd, const data_set& data,
  const glm_model& model) {
  // TODO
  return Rcpp::List();
}

#endif
