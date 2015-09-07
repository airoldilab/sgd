#ifndef MODEL_M_MODEL_H
#define MODEL_M_MODEL_H

#include "basedef.h"
#include "data/data_point.h"
#include "model/base_model.h"
#include "model/m-estimation/m_loss.h"

class m_model : public base_model {
  /**
   * M-estimation
   *
   * @param model attributes affiliated with model as R type
   */
public:
  m_model(Rcpp::List model) : base_model(model) {
    loss_ = Rcpp::as<std::string>(model["loss"]);
    if (loss_ == "huber") {
      loss_obj_ = new huber_loss();
    } else {
      Rcpp::Rcout << "warning: loss not implemented yet" << std::endl;
    }
    lambda_ = 3.0; // default for huber loss
  }

  mat gradient(unsigned t, const mat& theta_old, const data_set& data)
    const {
    data_point data_pt = data.get_data_point(t);
    return (loss_obj_->first_derivative(
      data_pt.y - dot(data_pt.x, theta_old), lambda_) * data_pt.x).t() +
      lambda1*norm(theta_old, 1) + lambda2*norm(theta_old, 2);
  }

  std::string loss() const {
    return loss_;
  }

  // Functions for implicit update
  // rho'(y - x^T theta + ||x||^2 * ksi)
  double scale_factor(double ksi, const data_point& data_pt, const mat&
    theta_old, double normx) const {
    return loss_obj_->first_derivative(
      data_pt.y - dot(theta_old, data_pt.x) + normx * ksi, lambda_);
  }

  // d/d(ksi) rho'(y - x^T theta + ||x||^2 * ksi)
  double scale_factor_first_deriv(double ksi, const data_point& data_pt, const
    mat& theta_old, double normx) const {
    return loss_obj_->second_derivative(
      data_pt.y - dot(theta_old, data_pt.x) + normx * ksi, lambda_)*normx;
  }

  // d^2/d(ksi)^2 rho'(y - x^T theta + ||x||^2 * ksi)
  double scale_factor_second_deriv(double ksi, const data_point& data_pt, const
    mat& theta_old, double normx) const {
    return loss_obj_->third_derivative(
      data_pt.y - dot(theta_old, data_pt.x) + normx * ksi, lambda_)*normx*normx;
  }

private:
  std::string loss_;
  base_loss* loss_obj_;
  double lambda_;
};

#endif
