#ifndef DATA_DATA_POINT_H
#define DATA_DATA_POINT_H

#include <iostream>
#include <vector>
#include "basedef.h"

struct data_point {
  /**
   * Collection of an individual observation's covariates and response.
   *
   * @param x covariates for a single sample
   * @param y response value for a single sample
   */
  // Constructors
  data_point() : x(mat()), y(0) {}
  data_point(const mat& x, double y) : x(x), y(y) {}

  // Members
  mat x;
  double y;
};

#endif
