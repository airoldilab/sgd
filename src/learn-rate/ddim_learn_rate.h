#ifndef LEARN_RATE_DDIM_LEARN_RATE_H
#define LEARN_RATE_DDIM_LEARN_RATE_H

#include "basedef.h"
#include "data/data_point.h"
#include "learn-rate/base_learn_rate.h"
#include "learn-rate/learn_rate_value.h"

using namespace arma;

typedef boost::function<mat(const mat&, const data_point&, double)> grad_func_type;

class ddim_learn_rate : public base_learn_rate {
  /**
   * d-dimensional learning rate with parameter weight alpha and exponent c
   * adagrad: a=1, b=1, c=1/2, eta=1, eps=1e-6
   * d-dim: a=0, b=1, c=1, eta=1, eps=1e-6
   * rmsprop: a=gamma, b=1-gamma, c=1/2, eta=1, eps=1e-6
   */
public:
  // Constructors
  ddim_learn_rate(unsigned d, double eta_, double a_, double b_, double c_,
                  double eps_, const grad_func_type& gr) :
    Idiag(ones<vec>(d)), eta(eta_), a(a_), b(b_), c(c_), eps(eps_),
    grad_func(gr), v(2, d) {}

  // Operators
  virtual const learn_rate_value& learning_rate(const mat& theta_old, const
    data_point& data_pt, double offset, unsigned t, unsigned d) {
    mat Gi = grad_func(theta_old, data_pt, offset);
    for (unsigned i = 0; i < d; ++i) {
      Idiag.at(i) = a * Idiag.at(i) + b * pow(Gi.at(i, 0), 2);
    }

    for (unsigned i = 0; i < d; ++i) {
      if (std::abs(Idiag.at(i)) > 1e-8) {
        v.lr_mat.at(i, i) = eta / pow(Idiag.at(i) + eps, c);
      }
      else {
        v.lr_mat.at(i, i) = Idiag.at(i);
      }
    }
    return v;
  }

private:
  vec Idiag;
  double a;
  double b;
  double c;
  double eta;
  double eps;
  grad_func_type grad_func;
  learn_rate_value v;
};

#endif
