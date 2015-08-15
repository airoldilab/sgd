#ifndef MODEL_COX_MODEL_H
#define MODEL_COX_MODEL_H

#include "basedef.h"
#include "data/data_point.h"
#include "model/base_model.h"
#include <boost/shared_ptr.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/ref.hpp>
#include <iostream>

class cox_model : public base_model {
  /**
   * Cox proportional hazards model
   *
   * @param model attributes affiliated with model as R type
   */
public:
  cox_model(Rcpp::List model) : base_model(model) {}

  mat gradient(unsigned t, const mat& theta_old, const data_set& data)
    const {
    data_point data_pt = data.get_data_point(t);
    unsigned j = data_pt.idx;

    // assuming data points fail in order, i.e., risk set R_i={i,i+1,...,n}
    vec xi = exp(data.X * theta_old);
    vec h = zeros<vec>(j);
    double sum_xi = 0;
    for (int i = j-1; i < j; --i) {
      // h_i = d_i/sum(xi[i:n])
      if (i == j-1) {
        for (int k = i; k < data.n_samples; ++k) {
          sum_xi += xi(k);
        }
      } else {
        sum_xi += xi(i);
      }
      h(i) = data.Y(i)/sum_xi;
    }
    double r = data_pt.y - xi(j) * sum(h);
    return (r * data_pt.x).t();
  }

  // TODO
  bool rank;
};

#endif
