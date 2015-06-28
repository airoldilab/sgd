#ifndef MODEL_BASE_MODEL_H
#define MODEL_BASE_MODEL_H

#include "basedef.h"
#include "data/data_point.h"
#include "learn-rate/base_learn_rate.h"
#include <boost/shared_ptr.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/bind/bind.hpp>
#include <boost/ref.hpp>
#include <iostream>

typedef boost::function<mat(const mat&, const data_point&)> grad_func_type;

class base_model;
//TODO
template<typename MODEL>
class Get_grad_coeff;
template<typename MODEL>
class Implicit_fn;

class base_model {
  /**
   * Base class for models
   *
   * @param experiment list of attributes to take from R type
   */
public:
  std::string model_name;
  unsigned d;
  unsigned n_passes; // sgd.control
  std::string lr; // sgd.control
  mat start; // sgd.control
  double delta; // sgd.control
  double lambda1; // sgd.control
  double lambda2; // sgd.control
  bool convergence; // sgd.control
  Rcpp::List model_attrs; // this should be expanded per derived model, not here
  mat weights; // TODO glm-specific
  bool trace; // TODO glm-specific
  bool dev; // TODO glm-specific

  // Constructors
  base_model(Rcpp::List experiment) {
    model_name = Rcpp::as<std::string>(experiment["name"]);
    d = Rcpp::as<unsigned>(experiment["d"]);
    n_passes = Rcpp::as<unsigned>(experiment["npasses"]);
    lr = Rcpp::as<std::string>(experiment["lr"]);
    start = Rcpp::as<mat>(experiment["start"]);
    weights = Rcpp::as<mat>(experiment["weights"]);
    delta = Rcpp::as<double>(experiment["delta"]);
    lambda1 = Rcpp::as<double>(experiment["lambda1"]);
    lambda2 = Rcpp::as<double>(experiment["lambda2"]);
    trace = Rcpp::as<bool>(experiment["trace"]);
    dev = Rcpp::as<bool>(experiment["deviance"]);
    convergence = Rcpp::as<bool>(experiment["convergence"]);
    model_attrs = experiment["model.attrs"];
  }

  // Gradient
  mat gradient(const mat& theta_old, const data_point& data_pt) const;

  const learn_rate_value& learning_rate(const mat& theta_old, const
    data_point& data_pt, unsigned t) {
    return (*lr_obj_)(theta_old, data_pt, t);
  }

protected:
  base_learn_rate* lr_obj_;
};

template<typename MODEL>
class Get_grad_coeff {
  // Compute gradient coeff and its derivative for Implicit-SGD update
public:
  Get_grad_coeff(const MODEL& e, const data_point& d,
    const mat& t, double n) :
    model(e), data_pt(d), theta_old(t), normx(n) {}

  double operator() (double ksi) const {
    return data_pt.y-model.h_transfer(dot(theta_old, data_pt.x)
                     + normx * ksi);
  }

  double first_derivative (double ksi) const {
    return model.h_first_derivative(dot(theta_old, data_pt.x)
           + normx * ksi)*normx;
  }

  double second_derivative (double ksi) const {
    return model.h_second_derivative(dot(theta_old, data_pt.x)
             + normx * ksi)*normx*normx;
  }

  const MODEL& model;
  const data_point& data_pt;
  const mat& theta_old;
  double normx;
};

template<typename MODEL>
class Implicit_fn {
  // Root finding functor for Implicit-SGD update
public:
  typedef boost::math::tuple<double, double, double> tuple_type;

  Implicit_fn(double a, const Get_grad_coeff<MODEL>& get_grad) :
    at(a), g(get_grad) {}

  tuple_type operator() (double u) const {
    double value = u - at * g(u);
    double first = 1 + at * g.first_derivative(u);
    double second = at * g.second_derivative(u);
    tuple_type result(value, first, second);
    return result;
  }

  double at;
  const Get_grad_coeff<MODEL>& g;
};

#endif
