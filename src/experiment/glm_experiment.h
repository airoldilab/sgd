#ifndef EXPERIMENT_GLM_EXPERIMENT_H
#define EXPERIMENT_GLM_EXPERIMENT_H

#include "basedef.h"
#include "data/data_point.h"
#include "experiment/base_experiment.h"
#include "glm_family.h"
#include "glm_transfer.h"
#include "learn-rate/onedim_learn_rate.h"
#include "learn-rate/onedim_eigen_learn_rate.h"
#include "learn-rate/ddim_learn_rate.h"
#include <boost/shared_ptr.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/bind/bind.hpp>
#include <boost/ref.hpp>
#include <iostream>

using namespace arma;

class glm_experiment : public base_experiment {
  /* Experiment class for generalized linear models */
public:
  glm_experiment(std::string m_name, Rcpp::List mp_attrs) :
    base_experiment(m_name, mp_attrs) {
    if (model_name == "gaussian") {
      family_ptr_type fp(new gaussian_family());
      family_obj_ = fp;
    }
    else if (model_name == "poisson") {
      family_ptr_type fp(new poisson_family());
      family_obj_ = fp;
    }
    else if (model_name == "binomial") {
      family_ptr_type fp(new binomial_family());
      family_obj_ = fp;
    }
    else if (model_name == "gamma") {
      family_ptr_type fp(new gamma_family());
      family_obj_ = fp;
    }

    if (model_name == "gaussian" ||
        model_name == "poisson" ||
        model_name == "binomial" ||
        model_name == "gamma") {
      std::string transfer_name = Rcpp::as<std::string>(model_attrs["transfer.name"]);
      rank = Rcpp::as<bool>(model_attrs["rank"]);

      if (transfer_name == "identity") {
        transfer_ptr_type tp(new identity_transfer());
        transfer_obj_ = tp;
      }
      else if (transfer_name == "exp") {
        transfer_ptr_type tp(new exp_transfer());
        transfer_obj_ = tp;
      }
      else if (transfer_name == "inverse") {
        transfer_ptr_type tp(new inverse_transfer());
        transfer_obj_ = tp;
      }
      else if (transfer_name == "logistic") {
        transfer_ptr_type tp(new logistic_transfer());
        transfer_obj_ = tp;
      }
    } else if (model_name == "...") {
      // code here
    }
  }

  // Gradient
  mat gradient(const mat& theta_old, const data_point& data_pt,
    double offset) const {
    return ((data_pt.y - h_transfer(dot(data_pt.x, theta_old) + offset)) *
      data_pt.x).t() + lambda1*norm(theta_old, 1) + lambda2*norm(theta_old, 2);
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

  bool valideta(double eta) const {
    return transfer_obj_->valideta(eta);
  }

  double variance(double u) const {
    return family_obj_->variance(u);
  }

  double deviance(const mat& y, const mat& mu, const mat& wt) const {
    return family_obj_->deviance(y, mu, wt);
  }

  friend std::ostream& operator<<(std::ostream& os, const base_experiment& exprm) {
    os << "  Experiment:\n" << "    Model: " << exprm.model_name << "\n" <<
          "    Learning rate: " << exprm.lr << std::endl;
    return os;
  }

  bool rank;

private:
  grad_func_type create_grad_func_instance() {
    grad_func_type grad_func = boost::bind(&glm_experiment::gradient, this, _1, _2, _3);
    return grad_func;
  }

  typedef boost::shared_ptr<base_transfer> transfer_ptr_type;
  transfer_ptr_type transfer_obj_;

  typedef boost::shared_ptr<base_family> family_ptr_type;
  family_ptr_type family_obj_;
};

#endif
