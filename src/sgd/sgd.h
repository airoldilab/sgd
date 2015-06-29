#ifndef SGD_SGD_H
#define SGD_SGD_H

#include <iostream>
#include <vector>
#include "basedef.h"
#include "data/data_point.h"
#include "learn-rate/base_learn_rate.h"
#include "learn-rate/onedim_learn_rate.h"
#include "learn-rate/onedim_eigen_learn_rate.h"
#include "learn-rate/ddim_learn_rate.h"

class sgd {
  /**
   * Collection of values and functions for stochastic gradient descent.
   *
   * @param n_samples  number of data samples
   * @param n_features number of features
   * @param n_passes   number of passes of data
   * @param start      starting values for SGD
   * @param method     stochastic gradient method
   * @param delta
   * @param convergence
   * @param ti         timer for benchmarking how long to get each estimate
   * @param size       number of estimates to store log-uniformly
   */
public:
  // Constructors
  sgd(unsigned n_samples, unsigned n_features, unsigned n_passes,
    const mat& start, std::string method, double delta, bool convergence,
    const boost::timer& ti, unsigned size=100) :
    n_passes_(n_passes),
    estimates_(mat(n_features, size)),
    last_estimate_(start),
    times_(size),
    method_(method),
    delta_(delta),
    convergence_(convergence),
    ti_(ti),
    n_iter_(n_samples*n_passes),
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
  unsigned get_n_passes() const {
    return n_passes_;
  }
  mat get_estimates() const {
    return estimates_;
  }
  mat get_last_estimate() const {
    return last_estimate_;
  }
  std::string get_method() const {
    return method_;
  }
  double get_delta() const {
    return delta_;
  }
  vec get_times() const {
    return times_;
  }
  Mat<unsigned> get_pos() const {
    return pos_;
  }

  // Setters
  void set_learn_rate(std::string lr, vec lr_control, unsigned d,
    grad_func_type grad_func) {
    if (lr == "one-dim") {
      lr_obj_ = new onedim_learn_rate(lr_control(0), lr_control(1),
                                      lr_control(2), lr_control(3));
    } else if (lr == "one-dim-eigen") {
      lr_obj_ = new onedim_eigen_learn_rate(d, grad_func);
    } else if (lr == "d-dim") {
      lr_obj_ = new ddim_learn_rate(d, 1., 0., 1., 1.,
                                    lr_control(0), grad_func);
    } else if (lr == "adagrad") {
      lr_obj_ = new ddim_learn_rate(d, lr_control(0), 1., 1., .5,
                                    lr_control(1), grad_func);
    } else if (lr == "rmsprop") {
      lr_obj_ = new ddim_learn_rate(d, lr_control(0), lr_control(1),
                                    1-lr_control(1), .5, lr_control(2),
                                    grad_func);
    }
  }

  // Learning rate
  const learn_rate_value& learning_rate(const mat& theta_old, const
    data_point& data_pt, unsigned t) {
    return (*lr_obj_)(theta_old, data_pt, t);
  }

  // Operators
  sgd& operator=(const mat& theta_new) {
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
  unsigned n_passes_;
  mat estimates_;
  mat last_estimate_;
  vec times_;
  std::string method_;
  double delta_;
  bool convergence_;
  boost::timer ti_;
  base_learn_rate* lr_obj_;
  unsigned n_iter_;     // total number of iterations
  unsigned iter_;       // current iteration
  unsigned size_;       // number of coefs to be recorded
  unsigned n_recorded_; // number of coefs that have been recorded
  Mat<unsigned> pos_;   // the iteration of recorded coefficients
};

#endif
