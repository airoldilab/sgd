#ifndef MODEL_BASE_MODEL_H
#define MODEL_BASE_MODEL_H

#include "basedef.h"
#include "data/data_point.h"
#include <boost/shared_ptr.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/ref.hpp>
#include <iostream>

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
   * @param model attributes affiliated with model as R type
   */
public:
  base_model(Rcpp::List model) {
    name_ = Rcpp::as<std::string>(model["name"]);
    lambda1 = Rcpp::as<double>(model["lambda1"]);
    lambda2 = Rcpp::as<double>(model["lambda2"]);
  }

  std::string name() const {
    return name_;
  }

  mat gradient(unsigned t, const mat& theta_old, const data_set& data) const;

  // TODO make private
  double lambda1;
  double lambda2;

protected:
  std::string name_;
};

template<typename MODEL>
class Get_grad_coeff {
  // Compute gradient coeff and its derivative for Implicit-SGD update
public:
  Get_grad_coeff(const MODEL& e, const data_point& d, const mat& t,
    double n) : model(e), data_pt(d), theta_old(t), normx(n) {}

  double operator()(double ksi) const {
    return data_pt.y-model.h_transfer(dot(theta_old, data_pt.x)
                     + normx * ksi);
  }

  double first_derivative(double ksi) const {
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

  tuple_type operator()(double u) const {
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
