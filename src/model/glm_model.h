#ifndef MODEL_GLM_MODEL_H
#define MODEL_GLM_MODEL_H

#include "basedef.h"
#include "data/data_point.h"
#include "model/base_model.h"
#include "model/glm/glm_family.h"
#include "model/glm/glm_transfer.h"
#include "learn-rate/onedim_learn_rate.h"
#include "learn-rate/onedim_eigen_learn_rate.h"
#include "learn-rate/ddim_learn_rate.h"
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
   * @param experiment list of attributes to take from R type
   */
public:
  glm_model(Rcpp::List experiment) :
    base_model(experiment) {
    vec lr_control= Rcpp::as<vec>(experiment["lr.control"]);
    if (lr == "one-dim") {
      lr_obj_ = new onedim_learn_rate(lr_control(0), lr_control(1),
                                      lr_control(2), lr_control(3));
    } else if (lr == "one-dim-eigen") {
      lr_obj_ = new onedim_eigen_learn_rate(d, grad_func());
    } else if (lr == "d-dim") {
      lr_obj_ = new ddim_learn_rate(d, 1., 0., 1., 1.,
                                    lr_control(0), grad_func());
    } else if (lr == "adagrad") {
      lr_obj_ = new ddim_learn_rate(d, lr_control(0), 1., 1., .5,
                                    lr_control(1), grad_func());
    } else if (lr == "rmsprop") {
      lr_obj_ = new ddim_learn_rate(d, lr_control(0), lr_control(1),
                                    1-lr_control(1), .5, lr_control(2),
                                    grad_func());
    }
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
    os << "  Model:\n" << "    Model name: " << exprm.model_name << "\n" <<
          "    Learning rate: " << exprm.lr << std::endl;
    return os;
  }

  bool rank;

private:
  base_transfer* transfer_obj_;
  base_family* family_obj_;
};

#endif
