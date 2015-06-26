#ifndef EXPERIMENT_BASE_EXPERIMENT_H
#define EXPERIMENT_BASE_EXPERIMENT_H

#include "basedef.h"
#include "data/data_point.h"
#include "learn-rate/onedim_learn_rate.h"
#include "learn-rate/onedim_eigen_learn_rate.h"
#include "learn-rate/ddim_learn_rate.h"
#include <boost/shared_ptr.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/bind/bind.hpp>
#include <boost/ref.hpp>
#include <iostream>

using namespace arma;

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
  mat offset;
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
  mat gradient(const mat& theta_old, const data_point& data_pt, double offset) const;

  // Learning rates
  void init_one_dim_learning_rate(double gamma, double alpha, double c,
    double scale) {
    learnrate_ptr_type lp(new onedim_learn_rate(gamma, alpha, c, scale));
    lr_obj_ = lp;
  }

  void init_one_dim_eigen_learning_rate() {
    grad_func_type grad_func = create_grad_func_instance();
    learnrate_ptr_type lp(new onedim_eigen_learn_rate(grad_func));
    lr_obj_ = lp;
  }

  void init_ddim_learning_rate(double eta, double a, double b, double c,
    double eps) {
    grad_func_type grad_func = create_grad_func_instance();
    learnrate_ptr_type lp(new ddim_learn_rate(d, eta, a, b, c, eps, grad_func));
    lr_obj_ = lp;
  }

  const learn_rate_value& learning_rate(const mat& theta_old, const
    data_point& data_pt, double offset, unsigned t) const {
    //return lr_(theta_old, data_pt, offset, t, d);
    return lr_obj_->learning_rate(theta_old, data_pt, offset, t, d);
  }

protected:
  virtual grad_func_type create_grad_func_instance() {
    grad_func_type grad_func;
    return grad_func;
  }

  typedef boost::shared_ptr<base_learn_rate> learnrate_ptr_type;
  learnrate_ptr_type lr_obj_;
};

template<typename EXPERIMENT>
class Get_grad_coeff {
  // Compute gradient coeff and its derivative for Implicit-SGD update
public:
  Get_grad_coeff(const EXPERIMENT& e, const data_point& d,
    const mat& t, double n, double off) :
    experiment(e), data_pt(d), theta_old(t), normx(n), offset(off) {}

  double operator() (double ksi) const {
    return data_pt.y-experiment.h_transfer(dot(theta_old, data_pt.x)
                     + normx * ksi +offset);
  }

  double first_derivative (double ksi) const {
    return experiment.h_first_derivative(dot(theta_old, data_pt.x)
           + normx * ksi + offset)*normx;
  }

  double second_derivative (double ksi) const {
    return experiment.h_second_derivative(dot(theta_old, data_pt.x)
             + normx * ksi + offset)*normx*normx;
  }

  const EXPERIMENT& experiment;
  const data_point& data_pt;
  const mat& theta_old;
  double normx;
  double offset;
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
