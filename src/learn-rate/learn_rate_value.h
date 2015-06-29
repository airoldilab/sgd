#ifndef LEARN_RATE_LEARN_RATE_VALUE_H
#define LEARN_RATE_LEARN_RATE_VALUE_H

#include "basedef.h"

class learn_rate_value {
  /**
   * Object to return for all learning rate classes
   *
   * @param t type of the value; 0 is scalar, 1 is vector, 2 is matrix
   * @param d dimension of parameters
   */
public:
  learn_rate_value(unsigned type, unsigned d) : type_(type) {
    if (type_ == 0) {
      lr_scalar_ = 1;
    } else if (type_ == 1) {
      lr_vector_ = ones<vec>(d);
    } else {
      lr_matrix_ = eye<mat>(d, d);
    }
  }

  // Getters
  double& at(unsigned i) {
    if (type_ == 1) {
      return lr_vector_.at(i);
    } else if (type_ == 2) {
      return lr_matrix_.at(i);
    } else {
      Rcpp::Rcout <<
        "Indexing vector/matrix entry when learning rate type is neither" <<
        std::endl;
      return lr_scalar_;
    }
  }

  const double& at(unsigned i) const {
    return at(i);
  }

  double& at(unsigned i, unsigned j) {
    if (type_ == 2) {
      return lr_matrix_.at(i, j);
    } else {
      Rcpp::Rcout <<
        "Indexing matrix entry when learning rate type is not matrix" <<
        std::endl;
      return lr_scalar_;
    }
  }

  const double& at(unsigned i, unsigned j) const {
    return at(i, j);
  }

  // Take average for usage in implicit SGD
  double mean() const {
    double average = 0.0;
    if (type_ == 0) {
      return lr_scalar_;
    } else if (type_ == 1) {
      return arma::mean(lr_vector_);
    } else {
      //return arma::mean(arma::mean(lr_matrix_));
      return arma::mean(lr_matrix_.diag());
    }
  }

  // Operators
  learn_rate_value operator=(double scalar) {
    if (type_ == 0) {
      lr_scalar_ = scalar;
    } else {
      Rcpp::Rcout <<
        "Setting learning rate value to scalar when its type is not" <<
        std::endl;
    }
    return *this;
  }

  learn_rate_value operator=(const vec& vector) {
    if (type_ == 1) {
      lr_vector_ = vector;
    } else {
      Rcpp::Rcout <<
        "Setting learning rate value to vector when its type is not" <<
        std::endl;
    }
    return *this;
  }

  learn_rate_value operator=(const mat& matrix) {
    if (type_ == 2) {
      lr_matrix_ = matrix;
    } else {
      Rcpp::Rcout <<
        "Setting learning rate value to matrix when its type is not" <<
        std::endl;
    }
    return *this;
  }

  mat operator*(const mat& matrix) {
    if (type_ == 0) {
      return lr_scalar_ * matrix;
    } else if (type_ == 1) {
      //int m = matrix.n_rows;
      ////int n = matrix.n_cols;
      //mat out = zeros<mat>(m, 1);
      //for (unsigned i = 0; i < m; ++i) {
      //  //for (unsigned j = 0; j < n; ++j) {
      //    //out.at(i) += lr_vector_.at(i) * matrix.at(i, 0);
      //  //}
      //  out.at(i, 0) = lr_vector_.at(i) * matrix.at(i, 0);
      //}
      //return out;
      //return diagmat(lr_vector_) * matrix;
      return mat(lr_vector_) % matrix;
    } else {
      return lr_matrix_ * matrix;
    }
  }

  bool operator<(const double thres) {
    if (type_ == 0) {
      return lr_scalar_ < thres;
    } else if (type_ == 1) {
      return all(lr_vector_ < thres);
    } else{
      return all(diagvec(lr_matrix_) < thres);
    }
  }

  bool operator>(const double thres) {
    return !(*this < thres);
  }

private:
  unsigned type_;
  double lr_scalar_;
  vec lr_vector_;
  mat lr_matrix_;
};

#endif
