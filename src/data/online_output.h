#ifndef DATA_ONLINE_OUTPUT_H
#define DATA_ONLINE_OUTPUT_H

#include <iostream>
#include <vector>
#include "basedef.h"
#include "data/data_set.h"

class online_output {
  /**
   * Collection of SGD-related values for the data set.
   *
   * @param data data set
   * @param init starting values for SGD
   * @param ti   timer for benchmarking how long to get each estimate
   * @param size number of estimates to store log-uniformly
   */
public:
  online_output(const data_set& data, const mat& init, const boost::timer& ti,
    unsigned n_passes, unsigned size=100) :
    estimates_(mat(data.n_features, size)),
    last_estimate_(init),
    times_(size),
    ti_(ti),
    n_iter_(data.n_samples*n_passes),
    iter_(0),
    size_(size),
    n_recorded_(0),
    pos_(Mat<unsigned>(1, size)) {
      for (unsigned i = 0; i < size_; ++i) {
        pos_(0, i) = int(round(pow(10, i * log10(n_iter_) / (size_-1))));
      }
      if (pos_(0, pos_.n_cols-1) != n_iter_) {
        pos_(0, pos_.n_cols-1) = n_iter_;
      }
      if (n_iter_ < size_) {
        Rcpp::Rcout << "Warning: Too few data points for plotting!" << std::endl;
      }
    }

  // Getters
  mat get_estimates() const {
    return estimates_;
  }
  mat get_last_estimate() const {
    return last_estimate_;
  }
  vec get_times() const {
    return times_;
  }
  Mat<unsigned> get_pos() const {
    return pos_;
  }

  online_output& operator=(const mat& theta_new) {
    last_estimate_ = theta_new;
    iter_ += 1;
    if (iter_ == pos_[n_recorded_]) {
      estimates_.col(n_recorded_) = theta_new;
      times_.at(n_recorded_) = ti_.elapsed();
      n_recorded_ += 1;
      while (n_recorded_ < size_ && pos_[n_recorded_-1] == pos_[n_recorded_]) {
        estimates_.col(n_recorded_) = theta_new;
        times_.at(n_recorded_) = times_.at(n_recorded_-1);
        n_recorded_ += 1;
      }
    }
    return *this;
  }

private:
  mat estimates_;
  mat last_estimate_;
  vec times_;
  boost::timer ti_;
  unsigned n_iter_;     // total number of iterations
  unsigned iter_;       // current iteration
  unsigned size_;       // number of coefs to be recorded
  unsigned n_recorded_; // number of coefs that have been recorded
  Mat<unsigned> pos_;   // the iteration of recorded coefficients
};

#endif
