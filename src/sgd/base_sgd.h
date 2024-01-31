#ifndef SGD_BASE_SGD_H
#define SGD_BASE_SGD_H

#include "../basedef.h"
#include "../learn-rate/base_learn_rate.h"
#include "../learn-rate/onedim_learn_rate.h"
#include "../learn-rate/onedim_eigen_learn_rate.h"
#include "../learn-rate/ddim_learn_rate.h"

class base_sgd {
  /**
   * Base class for stochastic gradient descent
   *
   * @param sgd       attributes affiliated with sgd as R type
   * @param n_samples number of data samples
   * @param ti        timer for benchmarking how long to get each estimate
   */
public:
  base_sgd(Rcpp::List sgd, unsigned n_samples, const boost::timer::cpu_timer& ti) : ti_(ti) {
    name_ = Rcpp::as<std::string>(sgd["method"]);
    n_params_ = Rcpp::as<unsigned>(sgd["nparams"]);
    reltol_ = Rcpp::as<double>(sgd["reltol"]);
    n_passes_ = Rcpp::as<unsigned>(sgd["npasses"]);
    size_ = Rcpp::as<unsigned>(sgd["size"]);
    estimates_ = zeros<mat>(n_params_, size_);
    last_estimate_ = Rcpp::as<mat>(sgd["start"]);
    times_ = zeros<vec>(size_);
    t_ = 0;
    n_recorded_ = 0;
    pos_ = Mat<unsigned>(1, size_);
    pass_ = Rcpp::as<bool>(sgd["pass"]);
    verbose_ = Rcpp::as<bool>(sgd["verbose"]);

    check_ = Rcpp::as<bool>(sgd["check"]);
    if (check_) {
      truth_ = Rcpp::as<mat>(sgd["truth"]);
    }

    // Set which iterations to store estimates
    unsigned n_iters = n_samples*n_passes_;
    for (unsigned i = 0; i < size_; ++i) {
      pos_(0, i) = int(round(pow(10.,
                   i * log10(static_cast<double>(n_iters)) / (size_-1))));
    }
    if (pos_(0, pos_.n_cols-1) != n_iters) {
      pos_(0, pos_.n_cols-1) = n_iters;
    }
    if (n_iters < size_) {
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
  vec get_times() const {
    return times_;
  }
  Mat<unsigned> get_pos() const {
    return pos_;
  }
  bool pass() const {
    return pass_;
  }
  bool verbose() const {
    return verbose_;
  }

  // Check if satisfy convergence threshold.
  bool check_convergence(const mat& theta_new, const mat& theta_old) const {
    // if checking against truth
    double diff;
    if (check_) {
      diff = mean(mean(pow(theta_new - truth_, 2)));
      if (diff < 0.001) {
        return true;
      }
    // if not running fixed number of iterations
    } else if (!pass_) {
      diff = mean(mean(abs(theta_new - theta_old))) /
        mean(mean(abs(theta_old)));
      if (diff < reltol_) {
        return true;
      }
    }
    return false;
  }

  const learn_rate_value& learning_rate(unsigned t, const mat& grad_t) {
    return (*lr_obj_)(t, grad_t);
  }

  //TODO declare update method
  //template<typename MODEL>
  //mat update(unsigned t, const mat& theta_old, const data_set& data,
  //MODEL& model, bool& good_gradient);

  // base_sgd& operator=(const mat& theta_new) {
  //   last_estimate_ = theta_new;
  //   t_ += 1;
  //   if (t_ == pos_[n_recorded_]) {
  //     estimates_.col(n_recorded_) = theta_new;
  //     times_.at(n_recorded_) = ti_.elapsed();
  //     n_recorded_ += 1;
  //     while (n_recorded_ < size_ && pos_[n_recorded_-1] == pos_[n_recorded_]) {
  //       estimates_.col(n_recorded_) = theta_new;
  //       times_.at(n_recorded_) = times_.at(n_recorded_-1);
  //       n_recorded_ += 1;
  //     }
  //   }
  //   return *this;
  // }
  
  base_sgd& operator=(const mat& theta_new) {
    last_estimate_ = theta_new;
    t_ += 1;
    if (t_ == pos_[n_recorded_]) {
      estimates_.col(n_recorded_) = theta_new;
      boost::timer::cpu_times times = ti_.elapsed();
      boost::chrono::nanoseconds wall_ns(times.wall);
      boost::chrono::nanoseconds user_ns(times.user);
      boost::chrono::nanoseconds system_ns(times.system);
      boost::chrono::duration<double> wall_sec = boost::chrono::duration_cast<boost::chrono::duration<double>>(wall_ns);
      boost::chrono::duration<double> user_sec = boost::chrono::duration_cast<boost::chrono::duration<double>>(user_ns);
      boost::chrono::duration<double> system_sec = boost::chrono::duration_cast<boost::chrono::duration<double>>(system_ns);
      // boost::chrono::duration<double> seconds = boost::chrono::duration_cast<boost::chrono::duration<double>>(boost::chrono::nanoseconds(times.user + times.system));
      times_.at(n_recorded_) = wall_sec.count() + user_sec.count() + system_sec.count();
      n_recorded_ += 1;
      while (n_recorded_ < size_ && pos_[n_recorded_-1] == pos_[n_recorded_]) {
        estimates_.col(n_recorded_) = theta_new;
        times_.at(n_recorded_) = times_.at(n_recorded_-1);
        n_recorded_ += 1;
      }
    }
    return *this;
  }

  void end_early() {
    // Throw away the space for things that were not recorded.
    pos_.shed_cols(n_recorded_, size_-1);
    estimates_.shed_cols(n_recorded_, size_-1);
    times_.shed_rows(n_recorded_, size_-1);
  }

protected:
  std::string name_;        // name of stochastic gradient method
  unsigned n_params_;       // number of parameters
  double reltol_;           // relative tolerance for convergence
  unsigned n_passes_;       // number of passes over data
  unsigned size_;           // number of estimates to be recorded (log-uniformly)
  mat estimates_;           // collection of stored estimates
  mat last_estimate_;       // last SGD estimate
  vec times_;               // times to reach each stored estimate
  boost::timer::cpu_timer ti_;         // timer
  base_learn_rate* lr_obj_; // learning rate
  unsigned t_;              // current iteration
  unsigned n_recorded_;     // number of coefs that have been recorded
  Mat<unsigned> pos_;       // the iteration of recorded coefficients
  bool pass_;               // whether to force running for n_passes_ over data
  bool verbose_;
  bool check_;
  mat truth_;
};

#endif
