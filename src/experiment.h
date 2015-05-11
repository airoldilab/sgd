#ifndef EXPERIMENT_H
#define EXPERIMENT_H
#define BOOST_DISABLE_ASSERTS true


#include "basedef.h"
#include "data.h"
#include "family.h"
#include "learningrate.h"
#include "transfer.h"
#include <boost/shared_ptr.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/bind/bind.hpp>
#include <boost/ref.hpp>
#include <iostream>

using namespace arma;

struct Sgd_Experiment;
//TODO
//struct Get_score_coeff;
//struct Implicit_fn;

// Base experiment class for arbitrary model
struct Sgd_Experiment {
//@members
  unsigned d;
  unsigned n_iters;
  std::string model_name;
  Rcpp::List model_attrs;
  std::string lr;
  mat offset;
  mat weights;
  mat start;
  double epsilon;
  bool trace;
  bool dev;
  bool convergence;

//@methods
  Sgd_Experiment(std::string m_name, Rcpp::List mp_attrs)
  : model_name(m_name), model_attrs(mp_attrs) {
  }

  void init_one_dim_learning_rate(double gamma, double alpha, double c, double scale);
  void init_one_dim_eigen_learning_rate();
  void init_ddim_learning_rate(double alpha, double c);
  mat learning_rate(const mat& theta_old, const Sgd_DataPoint& data_pt, double offset, unsigned t) const;
  mat score_function(const mat& theta_old, const Sgd_DataPoint& datapoint, double offset) const;
};


// Experiment class for arbitrary model
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

  void init_one_dim_learning_rate(double gamma, double alpha, double c, double scale) {
    learnrate_ptr_type lp(new Sgd_Onedim_Learn_Rate(gamma, alpha, c, scale));
    lr_obj_ = lp;

    lr = "One-dimensional learning rate";
  }

  void init_one_dim_eigen_learning_rate() {
    score_func_type score_func = create_score_func_instance();

    learnrate_ptr_type lp(new Sgd_Onedim_Eigen_Learn_Rate(score_func));
    lr_obj_ = lp;

    lr = "One-dimensional eigenvalue learning rate";
  }

  void init_ddim_learning_rate(double alpha, double c) {
    score_func_type score_func = create_score_func_instance();

    learnrate_ptr_type lp(new Sgd_Ddim_Learn_Rate(d, alpha, c, score_func));
    lr_obj_ = lp;

    lr = "d-dimensional learning rate";
  }

  mat learning_rate(const mat& theta_old, const Sgd_DataPoint& data_pt, double offset, unsigned t) const {
    //return lr_(theta_old, data_pt, offset, t, d);
    return lr_obj_->learning_rate(theta_old, data_pt, offset, t, d);
  }

  mat score_function(const mat& theta_old, const Sgd_DataPoint& datapoint, double offset) const {
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
    os << "  Experiment:\n" << "    Family: " << exprm.model_name << "\n" <<
          //"    Transfer function: " << exprm.transfer_name <<  "\n" <<
          "    Learning rate: " << exprm.lr << "\n\n" <<
          "    Trace: " << (exprm.trace ? "On" : "Off") << "\n" <<
          "    Deviance: " << (exprm.dev ? "On" : "Off") << "\n" <<
          "    Convergence: " << (exprm.convergence ? "On" : "Off") << "\n" <<
          "    Epsilon: " << exprm.epsilon << "\n" << std::endl;
    return os;
  }

private:
  score_func_type create_score_func_instance() {
    score_func_type score_func = boost::bind(&Sgd_Experiment_Glm::score_function, this, _1, _2, _3);
    return score_func;
  }

  typedef boost::shared_ptr<Sgd_Transfer_Base> transfer_ptr_type;
  transfer_ptr_type transfer_obj_;

  typedef boost::shared_ptr<Sgd_Family_Base> family_ptr_type;
  family_ptr_type family_obj_;

  typedef boost::shared_ptr<Sgd_Learn_Rate_Base> learnrate_ptr_type;
  learnrate_ptr_type lr_obj_;
};

// Compute score function coeff and its derivative for Implicit-SGD update
template<typename EXPERIMENT>
struct Get_score_coeff {

  Get_score_coeff(const EXPERIMENT& e, const Sgd_DataPoint& d,
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

  Implicit_fn(double a, const Get_score_coeff<EXPERIMENT>& get_score): at(a), g(get_score){}
  tuple_type operator() (double u) const{
    double value = u - at * g(u);
    double first = 1 + at * g.first_derivative(u);
    double second = at * g.second_derivative(u);
    tuple_type result(value, first, second);
    return result;
  }

  double at;
  const Get_score_coeff<EXPERIMENT>& g;
};

#endif
