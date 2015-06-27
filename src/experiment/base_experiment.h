#ifndef EXPERIMENT_BASE_EXPERIMENT_H
#define EXPERIMENT_BASE_EXPERIMENT_H

#include "basedef.h"
#include "data/data_point.h"
#include "learn-rate/base_learn_rate.h"
#include <boost/shared_ptr.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/bind/bind.hpp>
#include <boost/ref.hpp>
#include <iostream>

using namespace arma;

typedef boost::function<mat(const mat&, const data_point&)> grad_func_type;

class base_experiment;
//TODO
template<typename EXPERIMENT>
class Get_grad_coeff;
template<typename EXPERIMENT>
class Implicit_fn;

class base_experiment {
  /**
   * Base class for experiments
   */
public:
  std::string model_name;
  unsigned n_iters;
  unsigned d;
  unsigned n_passes;
  std::string lr;
  mat start;
  mat weights;
  double delta;
  double lambda1;
  double lambda2;
  bool trace;
  bool dev;
  bool convergence;
  Rcpp::List model_attrs;

  // Constructors
  base_experiment(std::string m_name, Rcpp::List mp_attrs) :
    model_name(m_name), model_attrs(mp_attrs) {}

  // Gradient
  mat gradient(const mat& theta_old, const data_point& data_pt) const;

  // Learning rates
  void set_learn_rate(base_learn_rate* lr) {
    lr_ = lr;
  }
  grad_func_type grad_func();

  const learn_rate_value& learning_rate(const mat& theta_old, const
    data_point& data_pt, unsigned t) {
    return (*lr_)(theta_old, data_pt, t, d);
  }

protected:
  base_learn_rate* lr_;
};

template<typename EXPERIMENT>
class Get_grad_coeff {
  // Compute gradient coeff and its derivative for Implicit-SGD update
public:
  Get_grad_coeff(const EXPERIMENT& e, const data_point& d,
    const mat& t, double n) :
    experiment(e), data_pt(d), theta_old(t), normx(n) {}

  double operator() (double ksi) const {
    return data_pt.y-experiment.h_transfer(dot(theta_old, data_pt.x)
                     + normx * ksi);
  }

  double first_derivative (double ksi) const {
    return experiment.h_first_derivative(dot(theta_old, data_pt.x)
           + normx * ksi)*normx;
  }

  double second_derivative (double ksi) const {
    return experiment.h_second_derivative(dot(theta_old, data_pt.x)
             + normx * ksi)*normx*normx;
  }

  const EXPERIMENT& experiment;
  const data_point& data_pt;
  const mat& theta_old;
  double normx;
};

template<typename EXPERIMENT>
class Implicit_fn {
  // Root finding functor for Implicit-SGD update
public:
  typedef boost::math::tuple<double, double, double> tuple_type;

  Implicit_fn(double a, const Get_grad_coeff<EXPERIMENT>& get_grad) :
    at(a), g(get_grad) {}

  tuple_type operator() (double u) const {
    double value = u - at * g(u);
    double first = 1 + at * g.first_derivative(u);
    double second = at * g.second_derivative(u);
    tuple_type result(value, first, second);
    return result;
  }

  double at;
  const Get_grad_coeff<EXPERIMENT>& g;
};

#endif
