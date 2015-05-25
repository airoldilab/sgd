#ifndef EXPERIMENT_H
#define EXPERIMENT_H
#define BOOST_DISABLE_ASSERTS true

#include "basedef.h"
#include "data.h"
#include "glm-family.h"
#include "glm-transfer.h"
#include "learningrate.h"
#include <boost/shared_ptr.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/bind/bind.hpp>
#include <boost/ref.hpp>
#include <iostream>

using namespace arma;

struct Sgd_Experiment;
//TODO
//struct Get_grad_coeff;
//struct Implicit_fn;

// Base experiment class for arbitrary model
struct Sgd_Experiment {
//@members
  std::string model_name;
  unsigned n_iters;
  unsigned d;
  unsigned n_passes;
  std::string lr;
  mat start;
  mat weights;
  mat offset;
  double delta;
  bool trace;
  bool dev;
  bool convergence;
  Rcpp::List model_attrs;

//@methods
  Sgd_Experiment(std::string m_name, Rcpp::List mp_attrs)
  : model_name(m_name), model_attrs(mp_attrs) {
  }

  // Gradient
  mat gradient(const mat& theta_old, const Sgd_DataPoint& datapoint, double offset) const;

  /*
   * Learning rates
   */
  void init_one_dim_learning_rate(double gamma, double alpha, double c, double scale) {
    learnrate_ptr_type lp(new Sgd_Onedim_Learn_Rate(gamma, alpha, c, scale));
    lr_obj_ = lp;
  }

  void init_one_dim_eigen_learning_rate() {
    grad_func_type grad_func = create_grad_func_instance();
    learnrate_ptr_type lp(new Sgd_Onedim_Eigen_Learn_Rate(grad_func));
    lr_obj_ = lp;
  }

  void init_ddim_learning_rate(double a, double b, double c, double eps) {
    grad_func_type grad_func = create_grad_func_instance();
    learnrate_ptr_type lp(new Sgd_Ddim_Learn_Rate(d, a, b, c, eps, grad_func));
    lr_obj_ = lp;
  }

  const Sgd_Learn_Rate_Value& learning_rate(const mat& theta_old, const Sgd_DataPoint& data_pt, double offset, unsigned t) const {
    //return lr_(theta_old, data_pt, offset, t, d);
    return lr_obj_->learning_rate(theta_old, data_pt, offset, t, d);
  }

protected:
  virtual grad_func_type create_grad_func_instance() {
    grad_func_type grad_func;
    return grad_func;
  }

  typedef boost::shared_ptr<Sgd_Learn_Rate_Base> learnrate_ptr_type;
  learnrate_ptr_type lr_obj_;
};

// Experiment class for estimating equations
struct Sgd_Experiment_Ee : public Sgd_Experiment {
//@members

//@methods
  Sgd_Experiment_Ee(std::string m_name, Rcpp::List mp_attrs)
  : Sgd_Experiment(m_name, mp_attrs) {
    //gr = Rcpp::Function(mp_attrs["gr"]);
    // TODO
    // if model_attrs["wmatrix"] == NULL {
      int k = 5;
      wmatrix_ = eye<mat>(k, k);
    // } else {
    // wmatrix_ = model_attrs["wmatrix"];
    // }
  }

  // Gradient
  mat gradient(const mat& theta_old, const Sgd_DataPoint& datapoint, double offset) const {
    Rcpp::NumericVector r_theta_old = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(theta_old));
    Rcpp::NumericVector r_datapoint = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(datapoint.x)); // TODO include both x and y (?)
    //Rcpp::NumericMatrix r_out = gr(r_theta_old, r_datapoint);
    //Rcpp::NumericMatrix r_out = mp_attrs["gr"](r_theta_old, r_datapoint);
    // TODO don't know how to do this part
    Rcpp::NumericMatrix r_out;
    mat out = Rcpp::as<mat>(r_out);
    // TODO include weighting matrix
    return out;
  }

private:
  grad_func_type create_grad_func_instance() {
    grad_func_type grad_func = boost::bind(&Sgd_Experiment_Ee::gradient, this, _1, _2, _3);
    return grad_func;
  }

  mat wmatrix_;
  //Rcpp::Function gr;
  //TODO look into how optim calls its C function, maybe it stores it too
};

// Experiment class for generalized linear models
struct Sgd_Experiment_Glm : public Sgd_Experiment {
//@members

//@methods
  Sgd_Experiment_Glm(std::string m_name, Rcpp::List mp_attrs)
  : Sgd_Experiment(m_name, mp_attrs) {
    if (model_name == "gaussian") {
      family_ptr_type fp(new Sgd_Gaussian());
      family_obj_ = fp;
    }
    else if (model_name == "poisson") {
      family_ptr_type fp(new Sgd_Poisson());
      family_obj_ = fp;
    }
    else if (model_name == "binomial") {
      family_ptr_type fp(new Sgd_Binomial());
      family_obj_ = fp;
    }
    else if (model_name == "gamma") {
      family_ptr_type fp(new Sgd_Gamma());
      family_obj_ = fp;
    }

    if (model_name == "gaussian" || model_name == "poisson" || model_name == "binomial" || model_name == "gamma") {
      std::string transfer_name = Rcpp::as<std::string>(model_attrs["transfer.name"]);
      rank = Rcpp::as<bool>(model_attrs["rank"]);

      if (transfer_name == "identity") {
        transfer_ptr_type tp(new Sgd_Identity_Transfer());
        transfer_obj_ = tp;
      }
      else if (transfer_name == "exp") {
        transfer_ptr_type tp(new Sgd_Exp_Transfer());
        transfer_obj_ = tp;
      }
      else if (transfer_name == "inverse") {
        transfer_ptr_type tp(new Sgd_Inverse_Transfer());
        transfer_obj_ = tp;
      }
      else if (transfer_name == "logistic") {
        transfer_ptr_type tp(new Sgd_Logistic_Transfer());
        transfer_obj_ = tp;
      }
    } else if (model_name == "...") {
      // code here
    }
  }

  // Gradient
  mat gradient(const mat& theta_old, const Sgd_DataPoint& datapoint, double offset) const {
    return ((datapoint.y - h_transfer(dot(datapoint.x, theta_old) + offset)) *
      datapoint.x).t();
  }

  // TODO not all models have these methods
  double h_transfer(double u) const {
    return transfer_obj_->transfer(u);
    //return transfer_(u);
  }

  mat h_transfer(const mat& u) const {
    return transfer_obj_->transfer(u);
    // return mat_transfer_(u);
  }

  double g_link(double u) const {
    return transfer_obj_->link(u);
  }

  double h_first_derivative(double u) const {
    return transfer_obj_->first_derivative(u);
  }

  double h_second_derivative(double u) const {
    return transfer_obj_->second_derivative(u);
  }

  bool valideta(double eta) const{
    return transfer_obj_->valideta(eta);
  }

  double variance(double u) const {
    return family_obj_->variance(u);
  }

  double deviance(const mat& y, const mat& mu, const mat& wt) const {
    return family_obj_->deviance(y, mu, wt);
  }

  friend std::ostream& operator<<(std::ostream& os, const Sgd_Experiment& exprm) {
    os << "  Experiment:\n" << "    Model: " << exprm.model_name << "\n" <<
          "    Learning rate: " << exprm.lr << std::endl;
    return os;
  }

  bool rank;

private:
  grad_func_type create_grad_func_instance() {
    grad_func_type grad_func = boost::bind(&Sgd_Experiment_Glm::gradient, this, _1, _2, _3);
    return grad_func;
  }

  typedef boost::shared_ptr<Sgd_Transfer_Base> transfer_ptr_type;
  transfer_ptr_type transfer_obj_;

  typedef boost::shared_ptr<Sgd_Family_Base> family_ptr_type;
  family_ptr_type family_obj_;
};

// Compute gradient coeff and its derivative for Implicit-SGD update
template<typename EXPERIMENT>
struct Get_grad_coeff {

  Get_grad_coeff(const EXPERIMENT& e, const Sgd_DataPoint& d,
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

  const EXPERIMENT& experiment;
  const Sgd_DataPoint& datapoint;
  const mat& theta_old;
  double normx;
  double offset;
};

// Root finding functor for Implicit-SGD update
template<typename EXPERIMENT>
struct Implicit_fn {
  typedef boost::math::tuple<double, double, double> tuple_type;

  Implicit_fn(double a, const Get_grad_coeff<EXPERIMENT>& get_grad): at(a), g(get_grad) {}
  // tuple_type operator() (double u) const{
  //   double value = u - at * g(u);
  //   double first = 1 + at * g.first_derivative(u);
  //   double second = at * g.second_derivative(u);
  //   tuple_type result(value, first, second);
  //   return result;
  // }
  double operator() (double u) const{
    double value = u - at * g(u);
    return value;
  }

  double at;
  const Get_grad_coeff<EXPERIMENT>& g;
};

#endif
