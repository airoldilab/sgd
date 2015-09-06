#ifndef MODEL_GLM_MODEL_H
#define MODEL_GLM_MODEL_H

#include "basedef.h"
#include "data/data_point.h"
#include "model/base_model.h"
#include "model/glm/glm_family.h"
#include "model/glm/glm_transfer.h"

class glm_model : public base_model {
  /**
   * Generalized linear models
   *
   * @param model attributes affiliated with model as R type
   */
public:
  glm_model(Rcpp::List model) : base_model(model) {
    family_ = Rcpp::as<std::string>(model["family"]);
    if (family_ == "gaussian") {
      family_obj_ = new gaussian_family();
    } else if (family_ == "poisson") {
      family_obj_ = new poisson_family();
    } else if (family_ == "binomial") {
      family_obj_ = new binomial_family();
    } else if (family_ == "gamma") {
      family_obj_ = new gamma_family();
    } else {
      Rcpp::Rcout << "warning: model not implemented yet" << std::endl;
    }
    transfer_ = Rcpp::as<std::string>(model["transfer"]);
    if (transfer_ == "identity") {
      transfer_obj_ = new identity_transfer();
    } else if (transfer_ == "exp") {
      transfer_obj_ = new exp_transfer();
    } else if (transfer_ == "inverse") {
      transfer_obj_ = new inverse_transfer();
    } else if (transfer_ == "logistic") {
      transfer_obj_ = new logistic_transfer();
    }
  }

  mat gradient(unsigned t, const mat& theta_old, const data_set& data)
    const {
    data_point data_pt = data.get_data_point(t);
    return ((data_pt.y - h_transfer(dot(data_pt.x, theta_old))) *
      data_pt.x).t() + lambda1*norm(theta_old, 1) + lambda2*norm(theta_old, 2);
  }

  double h_transfer(double u) const {
    return transfer_obj_->transfer(u);
  }

  mat h_transfer(const mat& u) const {
    return transfer_obj_->transfer(u);
  }

  double g_link(double u) const {
    return transfer_obj_->link(u);
  }

  double h_first_deriv(double u) const {
    return transfer_obj_->first_derivative(u);
  }

  double h_second_deriv(double u) const {
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

  std::string family() const {
    return family_;
  }

  std::string transfer() const {
    return transfer_;
  }

  // Functions for implicit update
  // ell(x^T theta + ||x||^2 * ksi)
  double scale_factor(double ksi, const data_point& data_pt, const mat&
    theta_old, double normx) const {
    return data_pt.y - h_transfer(dot(theta_old, data_pt.x) + normx * ksi);
  }

  // d/d(ksi) ell(x^T theta + ||x||^2 * ksi)
  double scale_factor_first_deriv(double ksi, const data_point& data_pt, const
    mat& theta_old, double normx) const {
    return h_first_deriv(dot(theta_old, data_pt.x) + normx * ksi)*normx;
  }

  // d^2/d(ksi)^2 ell(x^T theta + ||x||^2 * ksi)
  double scale_factor_second_deriv(double ksi, const data_point& data_pt, const
    mat& theta_old, double normx) const {
    return h_second_deriv(dot(theta_old, data_pt.x) + normx * ksi)*normx*
      normx;
  }

private:
  std::string family_;
  std::string transfer_;
  base_family* family_obj_;
  base_transfer* transfer_obj_;
};

#endif
