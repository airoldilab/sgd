#ifndef POST_PROCESS_GLM_POST_PROCESS_H
#define POST_PROCESS_GLM_POST_PROCESS_H

#include "basedef.h"
#include "data/data_set.h"
#include "model/glm_model.h"
#include "sgd/sgd.h"
#include <stdlib.h>

Rcpp::List post_process(const sgd& sgd_out, const data_set& data,
  const glm_model& model, mat& coef, unsigned X_rank) {
  // Check the validity of eta for all observations.
  if (!data.big) {
    mat eta;
    eta = data.X * sgd_out.get_last_estimate();
    mat mu;
    mu = model.h_transfer(eta);
    for (int i = 0; i < eta.n_rows; ++i) {
        if (!is_finite(eta[i])) {
          Rcpp::Rcout << "warning: NaN or non-finite eta" << std::endl;
          break;
        }
        if (!model.valideta(eta[i])) {
          Rcpp::Rcout << "warning: eta is not in the support" << std::endl;
          break;
        }
    }

    // Check the validity of mu for Poisson and Binomial family.
    double eps = 10. * datum::eps;
    if (model.name == "poisson") {
      if (any(vectorise(mu) < eps)) {
        Rcpp::Rcout << "warning: sgd.fit: fitted rates numerically 0 occurred" << std::endl;
      }
    } else if (model.name == "binomial") {
      if (any(vectorise(mu) < eps) or any(vectorise(mu) > (1-eps))) {
        Rcpp::Rcout << "warning: sgd.fit: fitted rates numerically 0 occurred" << std::endl;
      }
    }

    // Calculate the deviance.
    double dev = model.deviance(data.Y, mu, model.weights);

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
  } else {
    return Rcpp::List::create(
      Rcpp::Named("mu") = 0,
      Rcpp::Named("eta") = 0,
      Rcpp::Named("rank") = X_rank,
      Rcpp::Named("deviance") = 0);
  }
}

#endif
