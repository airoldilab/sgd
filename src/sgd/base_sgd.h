#ifndef SGD_BASE_SGD_H
#define SGD_BASE_SGD_H

#include "basedef.h"
#include "data/data_point.h"
#include "learn-rate/base_learn_rate.h"
#include "learn-rate/onedim_learn_rate.h"
#include "learn-rate/onedim_eigen_learn_rate.h"
#include "learn-rate/ddim_learn_rate.h"
#include <iostream>
#include <vector>

class base_sgd {
  /**
   * Base class for stochastic gradient descent
   *
   * @param sgd       attributes affiliated with sgd as R type
   * @param n_samples number of data samples
   * @param ti        timer for benchmarking how long to get each estimate
   */
public:
  base_sgd(Rcpp::List sgd, unsigned n_samples, const boost::timer& ti) : ti_(ti) {
    name_ = Rcpp::as<std::string>(sgd["method"]);
    n_params_ = Rcpp::as<unsigned>(sgd["nparams"]);
    n_passes_ = Rcpp::as<unsigned>(sgd["npasses"]);
    n_iters_ = n_samples*n_passes_;
    size_ = Rcpp::as<unsigned>(sgd["size"]);
    estimates_ = mat(n_params_, size_);
    last_estimate_ = Rcpp::as<mat>(sgd["start"]);
    times_ = vec(size_);
    t_ = 0;
    n_recorded_ = 0;
    pos_ = Mat<unsigned>(1, size_);
    verbose_ = Rcpp::as<bool>(sgd["verbose"]);

    // Set which iterations to store estimates
    for (unsigned i = 0; i < size_; ++i) {
      pos_(0, i) = int(round(pow(10, i * log10(n_iters_) / (size_-1))));
    }
    if (pos_(0, pos_.n_cols-1) != n_iters_) {
      pos_(0, pos_.n_cols-1) = n_iters_;
    }
    if (n_iters_ < size_) {
      Rcpp::Rcout << "Warning: Too few data points for plotting!" << std::endl;
    }

    // Set learning rate
    std:: string lr = Rcpp::as<std::string>(sgd["lr"]);
    vec lr_control = Rcpp::as<vec>(sgd["lr.control"]);
    if (lr == "one-dim") {
      lr_obj_ = new onedim_learn_rate(lr_control(0), lr_control(1),
                                      lr_control(2), lr_control(3));
    } else if (lr == "one-dim-eigen") {
      lr_obj_ = new onedim_eigen_learn_rate(n_params_);
    } else if (lr == "d-dim") {
      lr_obj_ = new ddim_learn_rate(n_params_, 1., 0., 1., 1.,
                                    lr_control(0));
    } else if (lr == "adagrad") {
      lr_obj_ = new ddim_learn_rate(n_params_, lr_control(0), 1., 1., .5,
                                    lr_control(1));
    } else if (lr == "rmsprop") {
      lr_obj_ = new ddim_learn_rate(n_params_, lr_control(0), lr_control(1),
                                    1-lr_control(1), .5, lr_control(2));
    }
  }

  std::string name() const {
    return name_;
  }
  // TODO set naming convention properly
  unsigned get_n_passes() const {
    return n_passes_;
  }
  mat get_estimates() const {
    return estimates_;
  }
  mat get_last_estimate() const {
    return last_estimate_;
  }
  bool verbose() const {
    return verbose_;
  }
  vec get_times() const {
    return times_;
  }
  Mat<unsigned> get_pos() const {
    return pos_;
  }

  const learn_rate_value& learning_rate(unsigned t, const mat& grad_t) {
    return (*lr_obj_)(t, grad_t);
  }

  //TODO declare update method
  //template<typename MODEL>
  //mat update(const data_set& data, MODEL& model, bool& good_gradient);

  base_sgd& operator=(const mat& theta_new) {
    last_estimate_ = theta_new;
    t_ += 1;
    if (t_ == pos_[n_recorded_]) {
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

protected:
  std::string name_;        // name of stochastic gradient method
  unsigned n_params_;       // number of parameters
  unsigned n_passes_;       // number of passes over data
  unsigned n_iters_;        // total number of iterations
  unsigned size_;           // number of estimates to be recorded (log-uniformly)
  mat estimates_;           // collection of stored estimates
  mat last_estimate_;       // last SGD estimate
  vec times_;               // times to reach each stored estimate
  boost::timer ti_;         // timer
  base_learn_rate* lr_obj_; // learning rate
  unsigned t_;              // current iteration
  unsigned n_recorded_;     // number of coefs that have been recorded
  Mat<unsigned> pos_;       // the iteration of recorded coefficients
  bool verbose_;
};

#endif
