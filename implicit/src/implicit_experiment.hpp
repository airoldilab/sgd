#ifndef IMPLICIT_EXPERIMENT_HPP
#define IMPLICIT_EXPERIMENT_HPP

#include "implicit_basedef.hpp"
#include "implicit_data.hpp"
#include "implicit_family.hpp"
#include "implicit_learningrate.hpp"
#include "implicit_transfer.hpp"
#include <boost/math/tools/roots.hpp>
#include <boost/bind/bind.hpp>
#include <boost/ref.hpp>
#include <iostream>

using namespace arma;

struct Imp_Experiment;
struct Get_score_coeff;
struct Implicit_fn;

struct Imp_Experiment {
//@members
  unsigned p;
  unsigned n_iters;
  std::string model_name;
  std::string transfer_name;
  std::string lr_type;
  mat offset;
  mat weights;
  mat start;
  double epsilon;
  bool trace;
  bool dev;
  bool convergence;

//@methods
  Imp_Experiment(std::string m_name, std::string tr_name)
  :model_name(m_name), transfer_name(tr_name) {
    if (model_name == "gaussian") {
      variance_ = boost::bind(&Imp_Gaussian::variance, _1);
      deviance_ = boost::bind(&Imp_Gaussian::deviance, _1, _2, _3);
    }
    else if (model_name == "poisson") {
      variance_ = boost::bind(&Imp_Poisson::variance, _1);
      deviance_ = boost::bind(&Imp_Poisson::deviance, _1, _2, _3);
    }
    else if (model_name == "binomial") {
      variance_ = boost::bind(&Imp_Binomial::variance, _1);
      deviance_ = boost::bind(&Imp_Binomial::deviance, _1, _2, _3);
    }
    else if (model_name == "Gamma") {
      variance_ = boost::bind(&Imp_Gamma::variance, _1);
      deviance_ = boost::bind(&Imp_Gamma::deviance, _1, _2, _3);
    }

    if (transfer_name == "identity") {
      // transfer() 's been overloaded, have to specify the function signature
      transfer_ = boost::bind(static_cast<double (*)(double)>(
                      &Imp_Identity_Transfer::transfer), _1);
      mat_transfer_ = boost::bind(static_cast<mat (*)(const mat&)>(
                      &Imp_Identity_Transfer::transfer), _1);
      transfer_first_deriv_ = boost::bind(
                      &Imp_Identity_Transfer::first_derivative, _1);
      transfer_second_deriv_ = boost::bind(
                      &Imp_Identity_Transfer::second_derivative, _1);
      valideta_ = boost::bind(&Imp_Identity_Transfer::valideta, _1);
    }
    else if (transfer_name == "exp") {
      transfer_ = boost::bind(static_cast<double (*)(double)>(
                      &Imp_Exp_Transfer::transfer), _1);
      mat_transfer_ = boost::bind(static_cast<mat (*)(const mat&)>(
                      &Imp_Exp_Transfer::transfer), _1);
      transfer_first_deriv_ = boost::bind(
                      &Imp_Exp_Transfer::first_derivative, _1);
      transfer_second_deriv_ = boost::bind(
                      &Imp_Exp_Transfer::second_derivative, _1);
      valideta_ = boost::bind(&Imp_Exp_Transfer::valideta, _1);
    }
    else if (transfer_name == "logistic") {
      transfer_ = boost::bind(static_cast<double (*)(double)>(
                      &Imp_Logistic_Transfer::transfer), _1);
      mat_transfer_ = boost::bind(static_cast<mat (*)(const mat&)>(
                      &Imp_Logistic_Transfer::transfer), _1);
      transfer_first_deriv_ = boost::bind(
                      &Imp_Logistic_Transfer::first_derivative, _1);
      transfer_second_deriv_ = boost::bind(
                      &Imp_Logistic_Transfer::second_derivative, _1);
      valideta_ = boost::bind(&Imp_Logistic_Transfer::valideta, _1);
    }
    else if (transfer_name == "inverse") {
      transfer_ = boost::bind(static_cast<double (*)(double)>(
                      &Imp_Inverse_Transfer::transfer), _1);
      mat_transfer_ = boost::bind(static_cast<mat (*)(const mat&)>(
                      &Imp_Inverse_Transfer::transfer), _1);
      transfer_first_deriv_ = boost::bind(
                      &Imp_Inverse_Transfer::first_derivative, _1);
      transfer_second_deriv_ = boost::bind(
                      &Imp_Inverse_Transfer::second_derivative, _1);
      valideta_ = boost::bind(&Imp_Inverse_Transfer::valideta, _1);
    }
  }

  void init_uni_dim_learning_rate(double gamma, double alpha, double c, double scale) {
    lr_ = boost::bind(&Imp_Unidim_Learn_Rate::learning_rate, 
                      _1, _2, _3, _4, _5, gamma, alpha, c, scale);
    lr_type = "Uni-dimension learning rate";
  }

  void init_px_dim_learning_rate() {
    score_func_type score_func = boost::bind(&Imp_Experiment::score_function, this, _1, _2, _3);
    lr_ = boost::bind(&Imp_Pxdim_Learn_Rate::learning_rate,
                      _1, _2, _3, _4, _5, score_func);
    lr_type = "Px-dimension learning rate";
  }

  mat learning_rate(const mat& theta_old, const Imp_DataPoint& data_pt, double offset, unsigned t) const {
    //return lr(t);
    return lr_(theta_old, data_pt, offset, t, p);
  }

  mat score_function(const mat& theta_old, const Imp_DataPoint& datapoint, double offset) const {
    return ((datapoint.y - h_transfer(as_scalar(datapoint.x * theta_old))+offset)*datapoint.x).t();
  }

  double h_transfer(double u) const {
    return transfer_(u);
  }

  mat h_transfer(const mat& u) const {
    return mat_transfer_(u);
  }

  //YKuang
  double h_first_derivative(double u) const{
    return transfer_first_deriv_(u);
  }
  //YKuang
  double h_second_derivative(double u) const{
    return transfer_second_deriv_(u);
  }

  double variance(double u) const {
    return variance_(u);
  }

  double deviance(const mat& y, const mat& mu, const mat& wt) const {
    return deviance_(y, mu, wt);
  }

  bool valideta(double eta) const{
    return valideta_(eta);
  }

  friend std::ostream& operator<<(std::ostream& os, const Imp_Experiment& exprm) {
    os << "  Experiment:\n" << "    Family: " << exprm.model_name << "\n" <<
          "    Transfer function: " << exprm.transfer_name <<  "\n" <<
          "    Learning rate: " << exprm.lr_type << "\n\n" <<
          "    Trace: " << (exprm.trace ? "On" : "Off") << "\n" <<
          "    Deviance: " << (exprm.dev ? "On" : "Off") << "\n" <<
          "    Convergence: " << (exprm.convergence ? "On" : "Off") << "\n" <<
          "    Epsilon: " << exprm.epsilon << "\n" << std::endl;
    return os;
  }

private:
  uni_func_type transfer_;
  mmult_func_type mat_transfer_;
  uni_func_type transfer_first_deriv_;
  uni_func_type transfer_second_deriv_;

  learning_rate_type lr_;

  uni_func_type variance_;
  deviance_type deviance_;
  boost::function<bool (double)> valideta_;
};

// Compute score function coeff and its derivative for Implicit-SGD update
struct Get_score_coeff {

  //Get_score_coeff(const Imp_Experiment<TRANSFER>& e, const Imp_DataPoint& d,
  Get_score_coeff(const Imp_Experiment& e, const Imp_DataPoint& d,
      const mat& t, double n, double off) : experiment(e), datapoint(d), theta_old(t), normx(n), offset(off) {}

  double operator() (double ksi) const{
    return datapoint.y-experiment.h_transfer(dot(theta_old, datapoint.x)
                     + normx * ksi +offset);
  }

  double first_derivative (double ksi) const{
    return experiment.h_first_derivative(dot(theta_old, datapoint.x)
           + normx * ksi + offset)*normx;
  }

  double second_derivative (double ksi) const{
    return experiment.h_second_derivative(dot(theta_old, datapoint.x)
             + normx * ksi + offset)*normx*normx;
  }

  const Imp_Experiment& experiment;
  const Imp_DataPoint& datapoint;
  const mat& theta_old;
  double normx;
  double offset;
};

// Root finding functor for Implicit-SGD update
struct Implicit_fn {
  typedef boost::math::tuple<double, double, double> tuple_type;

  //Implicit_fn(double a, const Get_score_coeff<TRANSFER>& get_score): at(a), g(get_score){}
  Implicit_fn(double a, const Get_score_coeff& get_score): at(a), g(get_score){}
  tuple_type operator() (double u) const{
    double value = u - at * g(u);
    double first = 1 + at * g.first_derivative(u);
    double second = at * g.second_derivative(u);
    tuple_type result(value, first, second);
    return result;
  }
  
  double at;
  //const Get_score_coeff<TRANSFER>& g;
  const Get_score_coeff& g;
};

#endif