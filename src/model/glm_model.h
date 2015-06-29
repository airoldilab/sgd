#ifndef MODEL_GLM_MODEL_H
#define MODEL_GLM_MODEL_H

#include "basedef.h"
#include "data/data_point.h"
#include "model/base_model.h"
#include "model/glm/glm_family.h"
#include "model/glm/glm_transfer.h"
#include <boost/shared_ptr.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/bind/bind.hpp>
#include <boost/ref.hpp>
#include <iostream>

typedef boost::function<mat(const mat&, const data_point&)> grad_func_type;

class glm_model : public base_model {
  /**
   * Generalized linear models
   *
   * @param model attributes affiliated with model as R type
   */
public:
  // Constructors
  glm_model(Rcpp::List model) :
    base_model(model) {
    if (name == "gaussian") {
      family_obj_ = new gaussian_family();
    } else if (name == "poisson") {
      family_obj_ = new poisson_family();
    } else if (name == "binomial") {
      family_obj_ = new binomial_family();
    } else if (name == "gamma") {
      family_obj_ = new gamma_family();
    } else {
      Rcpp::Rcout << "warning: model not implemented yet" << std::endl;
    }
    if (name == "gaussian" ||
        name == "poisson" ||
        name == "binomial" ||
        name == "gamma") {
      Rcpp::List model_attrs = model["model.attrs"];
      std::string transfer_name = Rcpp::as<std::string>(model_attrs["transfer.name"]);
      rank = Rcpp::as<bool>(model_attrs["rank"]);
      weights = Rcpp::as<mat>(model_attrs["weights"]);
      trace = Rcpp::as<bool>(model_attrs["trace"]);
      dev = Rcpp::as<bool>(model_attrs["deviance"]);
      if (transfer_name == "identity") {
        transfer_obj_ = new identity_transfer();
      } else if (transfer_name == "exp") {
        transfer_obj_ = new exp_transfer();
      } else if (transfer_name == "inverse") {
        transfer_obj_ = new inverse_transfer();
      } else if (transfer_name == "logistic") {
        transfer_obj_ = new logistic_transfer();
      }
    }
  }

  // Gradient
  mat gradient(const mat& theta_old, const data_point& data_pt) const {
    return ((data_pt.y - h_transfer(dot(data_pt.x, theta_old))) *
      data_pt.x).t() + lambda1*norm(theta_old, 1) + lambda2*norm(theta_old, 2);
  }

  grad_func_type grad_func() {
    return boost::bind(&glm_model::gradient, this, _1, _2);
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

  friend std::ostream& operator<<(std::ostream& os, const base_model& exprm) {
    os << "  Model:\n" << "    Model name: " << exprm.name << std::endl;
    return os;
  }

  // Members
  mat weights;
  bool trace;
  bool dev;
  bool rank;

private:
  base_transfer* transfer_obj_;
  base_family* family_obj_;
};

#endif
