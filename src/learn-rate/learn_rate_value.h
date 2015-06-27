#ifndef LEARN_RATE_LEARN_RATE_VALUE_H
#define LEARN_RATE_LEARN_RATE_VALUE_H

#include "basedef.h"

class learn_rate_value {
  /* Object to return for all learning rate classes; it collects the return
   * value which can be a scalar, vector, or matrix. */
public:
  learn_rate_value(unsigned t, unsigned d) : type(t), dim(d) {
    if (type == 0) { // scalar
      lr_scalar = 1;
    }
    else if (type == 1) { // vector
      lr_vec = ones<vec>(d);
    }
    else { // matrix
      lr_mat = eye<mat>(d, d);
    }
  }

  mat lr_mat;
  vec lr_vec;
  double lr_scalar;
  unsigned type;
  unsigned dim;
};

// Overload operators so as to work for arbitrary learning rate value.
mat operator*(const learn_rate_value& lr, const mat& grad) {
  if (lr.type == 0) {
    return lr.lr_scalar * grad;
  } else if (lr.type == 1) {
    //int m = grad.n_rows;
    ////int n = grad.n_cols;
    //mat out = zeros<mat>(m, 1);
    //for (unsigned i = 0; i < m; ++i) {
    //  //for (unsigned j = 0; j < n; ++j) {
    //    //out.at(i) += lr.lr_vec.at(i) * grad.at(i, 0);
    //  //}
    //  out.at(i, 0) = lr.lr_vec.at(i) * grad.at(i, 0);
    //}
    //return out;
    //return diagmat(lr.lr_vec) * grad;
    return mat(lr.lr_vec) % grad;
  } else {
    return lr.lr_mat * grad;
  }
}

bool operator<(const learn_rate_value& lr, const double thres) {
  if (lr.type == 0) {
    return lr.lr_scalar < thres;
  } else if (lr.type == 1) {
    return all(lr.lr_vec < thres);
  } else{
    return all(diagvec(lr.lr_mat) < thres);
  }
}

bool operator>(const learn_rate_value& lr, const double thres) {
  return !(lr < thres);
}

std::ostream& operator<<(std::ostream& os, const learn_rate_value& lr) {
  if (lr.type == 0) {
    os << lr.lr_scalar;
  } else if (lr.type == 1) {
    os << lr.lr_vec;
  }
  else {
    os << lr.lr_mat;
  }
  return os;
}

#endif
