#ifndef LEARN_RATE_ONEDIM_LEARN_RATE_H
#define LEARN_RATE_ONEDIM_LEARN_RATE_H

#include "basedef.h"
#include "data/data_point.h"
#include "learn-rate/base_learn_rate.h"
#include "learn-rate/learn_rate_value.h"

using namespace arma;

class onedim_learn_rate : public base_learn_rate {
  /**
   * One-dimensional (scalar) learning rate, following Xu
   */
public:
  // Constructors
  onedim_learn_rate(double g, double a, double c_, double s) :
  gamma(g), alpha(a), c(c_), scale(s), v(0, 1) {}

  // Operators
  virtual const learn_rate_value& learning_rate(const mat& theta_old, const
    data_point& data_pt, double offset, unsigned t, unsigned d) {
    v.lr_scalar = scale * gamma * pow(1 + alpha * gamma * t, -c);
    return v;
  }

private:
  double gamma;
  double alpha;
  double c;
  double scale;
  learn_rate_value v;
};

#endif
