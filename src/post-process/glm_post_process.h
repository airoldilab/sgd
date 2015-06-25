#ifndef POST_PROCESS_GLM_POST_PROCESS_H
#define POST_PROCESS_GLM_POST_PROCESS_H

#include "basedef.h"
#include "data/data_set.h"
#include "data/online_output.h"
#include <stdlib.h>

template<typename EXPERIMENT>
Rcpp::List glm_post_process(const online_output& out, const data_set& data,
  const EXPERIMENT& exprm, mat& coef, unsigned X_rank) {
  // Check the validity of eta for all observations.
  if (!data.big) {
    mat eta;
    eta = data.X * out.get_last_estimate() + exprm.offset;
    mat mu;
    mu = exprm.h_transfer(eta);
    for (int i = 0; i < eta.n_rows; ++i) {
        if (!is_finite(eta[i])) {
          Rcpp::Rcout << "warning: NaN or non-finite eta" << std::endl;
          break;
        }
        if (!exprm.valideta(eta[i])) {
          Rcpp::Rcout << "warning: eta is not in the support" << std::endl;
          break;
        }
    }

    // Check the validity of mu for Poisson and Binomial family.
    double eps = 10. * datum::eps;
    if (exprm.model_name == "poisson")
      if (any(vectorise(mu) < eps))
        Rcpp::Rcout << "warning: sgd.fit: fitted rates numerically 0 occurred" << std::endl;
    if (exprm.model_name == "binomial")
        if (any(vectorise(mu) < eps) or any(vectorise(mu) > (1-eps)))
          Rcpp::Rcout << "warning: sgd.fit: fitted rates numerically 0 occurred" << std::endl;

    // Calculate the deviance.
    double dev = exprm.deviance(data.Y, mu, exprm.weights);

    // Check the number of features.
    if (X_rank < data.n_features) {
      for (int i = X_rank; i < coef.n_rows; i++) {
        coef.row(i) = datum::nan;
      }
    }
    return Rcpp::List::create(
      Rcpp::Named("mu") = mu,
      Rcpp::Named("eta") = eta,
      Rcpp::Named("rank") = X_rank,
      Rcpp::Named("deviance") = dev);
    }
  else{
    return Rcpp::List::create(
      Rcpp::Named("mu") = 0,
      Rcpp::Named("eta") = 0,
      Rcpp::Named("rank") = X_rank,
      Rcpp::Named("deviance") = 0);
  }
}

#endif
