#ifndef MODEL_M_MODEL_H
#define MODEL_M_MODEL_H

#include "../basedef.h"
#include "../data/data_point.h"
#include "base_model.h"
#include "m-estimation/m_loss.h"

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
      data_pt.y - dot(data_pt.x, theta_old), lambda_) * data_pt.x).t() -
      gradient_penalty(theta_old);
  }

  std::string loss() const {
    return loss_;
  }

  // Functions for implicit update
  double scale_factor(double ksi, double at, const data_point& data_pt, const
    mat& theta_old, double normx) const {
    return loss_obj_->first_derivative(
      data_pt.y - dot(theta_old, data_pt.x) -
        at * dot(gradient_penalty(theta_old), data_pt.x) +
        ksi * normx,
      lambda_);
  }

  double scale_factor_first_deriv(double ksi, double at, const data_point&
    data_pt, const mat& theta_old, double normx) const {
    return loss_obj_->second_derivative(
      data_pt.y - dot(theta_old, data_pt.x) -
        at * dot(gradient_penalty(theta_old), data_pt.x) +
        ksi * normx,
      lambda_) * normx;
  }

  double scale_factor_second_deriv(double ksi, double at, const data_point&
    data_pt, const mat& theta_old, double normx) const {
    return loss_obj_->third_derivative(
      data_pt.y - dot(theta_old, data_pt.x) -
        at * dot(gradient_penalty(theta_old), data_pt.x) +
        ksi * normx,
      lambda_) * normx * normx;
  }

private:
  std::string loss_;
  base_loss* loss_obj_;
  double lambda_;
};

#endif
