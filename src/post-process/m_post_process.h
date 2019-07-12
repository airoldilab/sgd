#ifndef POST_PROCESS_M_POST_PROCESS_H
#define POST_PROCESS_M_POST_PROCESS_H

#include "../basedef.h"
#include "../data/data_set.h"
#include "../model/m_model.h"

template <typename SGD>
Rcpp::List post_process(const SGD& sgd, const data_set& data,
  const m_model& model) {
  return Rcpp::List::create(
    Rcpp::Named("loss") = model.loss());
}

#endif
