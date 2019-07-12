#ifndef MODEL_M_LOSS_H
#define MODEL_M_LOSS_H

#include "../../basedef.h"

class base_loss;
class huber_loss;

class base_loss {
  /* Base class from which all loss function classes inherit from */
public:
  virtual double loss(double u, double lambda) const = 0;
  virtual double first_derivative(double u, double lambda) const = 0;
  virtual double second_derivative(double u, double lambda) const = 0;
  virtual double third_derivative(double u, double lambda) const = 0;
  virtual mat loss(const mat& u, double lambda) const {
    mat result = mat(u);
    for (unsigned i = 0; i < result.n_rows; ++i) {
      result(i, 0) = loss(u(i, 0), lambda);
    }
    return result;
  }
  virtual mat first_derivative(const mat& u, double lambda) const {
    mat result = mat(u);
    for (unsigned i = 0; i < result.n_rows; ++i) {
      result(i, 0) = first_derivative(u(i, 0), lambda);
    }
    return result;
  }
};

class huber_loss : public base_loss {
public:
  virtual double loss(double u, double lambda) const {
    if (std::abs(u) <= lambda) {
      return pow(u, 2)/2;
    } else {
      return lambda*std::abs(u) - pow(lambda, 2)/2;
    }
  }

  virtual double first_derivative(double u, double lambda) const {
    if (std::abs(u) <= lambda) {
      return u;
    } else {
      return lambda*sign(u);
    }
  }

  virtual double second_derivative(double u, double lambda) const {
    if (std::abs(u) <= lambda) {
      return 1.0;
    } else {
      return 0.0;
    }
  }

  virtual double third_derivative(double u, double lambda) const {
    return 0.0;
  }

private:
  template<typename T>
  double sign(const T& x) const {
    if (x > 0) {
      return 1.0;
    } else if (x < 0) {
      return -1.0;
    } else {
      return 0.0;
    }
  }
};

#endif
