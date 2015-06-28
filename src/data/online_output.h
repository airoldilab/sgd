#ifndef DATA_ONLINE_OUTPUT_H
#define DATA_ONLINE_OUTPUT_H

#include <iostream>
#include <vector>
#include "basedef.h"
#include "data/data_set.h"

class online_output {
  /* Collection of SGD-related values for the data set. */
public:
  online_output(const data_set& data, const mat& init, unsigned s=100) :
    estimates(mat(data.n_features, s)), initial(init), last_estimate(init),
    times(s), t(data.ti), n_iter(data.n_samples), iter(0), size(s),
    n_recorded(0), pos(Mat<unsigned>(1, s)) {
      for (unsigned i=0; i < size; ++i) {
        pos(0, i) = int(round(pow(10, i * log10(n_iter) / (size-1))));
      }
      if (pos(0, pos.n_cols-1) != n_iter)
        pos(0, pos.n_cols-1) = n_iter;
      if (n_iter < size)
        Rcpp::Rcout << "Warning: Too few data points for plotting!" << std::endl;
    }

  // Getters
  mat get_estimates() const {
    return estimates;
  }
  mat get_last_estimate() const {
    return last_estimate;
  }
  vec get_times() const {
    return times;
  }
  Mat<unsigned> get_pos() const {
    return pos;
  }

  online_output& operator=(const mat& theta_new) {
    last_estimate = theta_new;
    iter += 1;
    if (iter == pos[n_recorded]) {
      estimates.col(n_recorded) = theta_new;
      times.at(n_recorded) = t.elapsed();
      n_recorded += 1;
      while (n_recorded < size && pos[n_recorded-1] == pos[n_recorded]) {
        estimates.col(n_recorded) = theta_new;
        times.at(n_recorded) = times.at(n_recorded-1);
        n_recorded += 1;
      }
    }
    return *this;
  }

private:
  mat estimates;
  mat initial;
  mat last_estimate;
  vec times;
  boost::timer t;
  unsigned n_iter; // Total number of iterations
  unsigned iter; // Current iteration
  unsigned size; // Number of coefs to be recorded
  unsigned n_recorded; //Number of coefs that have been recorded
  Mat<unsigned> pos; //The iteration of recorded coefficients
};

#endif
