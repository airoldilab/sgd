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

typedef boost::function<mat(const mat&, const data_point&)> grad_func_type;

class glm_experiment : public base_experiment {
  /**
   * Generalized linear models
   */
public:
  glm_experiment(std::string m_name, Rcpp::List mp_attrs) :
    base_experiment(m_name, mp_attrs) {
    if (model_name == "gaussian") {
      family_obj_ = new gaussian_family();
    }
    else if (model_name == "poisson") {
      family_obj_ = new poisson_family();
    }
    else if (model_name == "binomial") {
      family_obj_ = new binomial_family();
    }
    else if (model_name == "gamma") {
      family_obj_ = new gamma_family();
    }

    if (model_name == "gaussian" ||
        model_name == "poisson" ||
        model_name == "binomial" ||
        model_name == "gamma") {
      std::string transfer_name = Rcpp::as<std::string>(model_attrs["transfer.name"]);
      rank = Rcpp::as<bool>(model_attrs["rank"]);
      if (transfer_name == "identity") {
        transfer_obj_ = new identity_transfer();
      }
      else if (transfer_name == "exp") {
        transfer_obj_ = new exp_transfer();
      }
      else if (transfer_name == "inverse") {
        transfer_obj_ = new inverse_transfer();
      }
      else if (transfer_name == "logistic") {
        transfer_obj_ = new logistic_transfer();
      }
    } else {
      Rcpp::Rcout << "Model not implemented yet " << std::endl;
    }
  }

  // Gradient
  mat gradient(const mat& theta_old, const data_point& data_pt) const {
    return ((data_pt.y - h_transfer(dot(data_pt.x, theta_old))) *
      data_pt.x).t() + lambda1*norm(theta_old, 1) + lambda2*norm(theta_old, 2);
  }

  grad_func_type grad_func() {
    return boost::bind(&glm_experiment::gradient, this, _1, _2);
  }

  // TODO not all models have these methods
  double h_transfer(double u) const {
    return transfer_obj_->transfer(u);
  }

  mat h_transfer(const mat& u) const {
    return transfer_obj_->transfer(u);
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
  base_transfer* transfer_obj_;
  base_family* family_obj_;
};

#endif
