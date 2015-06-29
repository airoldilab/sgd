#ifndef LEARN_RATE_DDIM_LEARN_RATE_H
#define LEARN_RATE_DDIM_LEARN_RATE_H

#include "basedef.h"
#include "data/data_point.h"
#include "learn-rate/base_learn_rate.h"
#include "learn-rate/learn_rate_value.h"

typedef boost::function<mat(const mat&, const data_point&)> grad_func_type;

class ddim_learn_rate : public base_learn_rate {
  /**
   * d-dimensional learning rate, which includes as special cases popular
   * learning rates:
   * adagrad: a=1, b=1, c=1/2, eta=1, eps=1e-6
   * d-dim: a=0, b=1, c=1, eta=1, eps=1e-6
   * rmsprop: a=gamma, b=1-gamma, c=1/2, eta=1, eps=1e-6
   *
   * @param d   dimension of learning rate
   * @param eta scale factor in numerator
   * @param a   factor to weigh old gradient information
   * @param b   factor to weigh new gradient information
   * @param c   power to exponentiate by
   * @param eps value to prevent division by zero
   * @param gr  gradient function
   */
public:
  // Constructors
  ddim_learn_rate(unsigned d, double eta, double a, double b, double c,
                  double eps, const grad_func_type& gr) :
    d_(d), Idiag_(ones<vec>(d)), eta_(eta), a_(a), b_(b), c_(c), eps_(eps),
    grad_func_(gr), v_(2, d) {} // TODO change to 1

  // Operators
  virtual const learn_rate_value& operator()(const mat& theta_old, const
    data_point& data_pt, unsigned t) {
    mat Gi = grad_func_(theta_old, data_pt);
    for (unsigned i = 0; i < d_; ++i) {
      Idiag_.at(i) = a_ * Idiag_.at(i) + b_ * pow(Gi.at(i, 0), 2);
    }

    for (unsigned i = 0; i < d_; ++i) {
      if (std::abs(Idiag_.at(i)) > 1e-8) {
        v_.at(i, i) = eta_ / pow(Idiag_.at(i) + eps_, c_);
      }
      else {
        v_.at(i, i) = Idiag_.at(i);
      }
    }
    return v_;
  }

private:
  unsigned d_;
  vec Idiag_;
  double a_;
  double b_;
  double c_;
  double eta_;
  double eps_;
  grad_func_type grad_func_;
  learn_rate_value v_;
};

#endif
