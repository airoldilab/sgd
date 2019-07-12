#ifndef DATA_DATA_POINT_H
#define DATA_DATA_POINT_H

#include "../basedef.h"

struct data_point {
  /**
   * Collection of an individual observation's covariates and response.
   *
   * @param x   covariates for a single sample
   * @param y   response value for a single sample
   * @param idx index of that data point into the data set
   */
  data_point(const mat& x, double y, unsigned idx) : x(x), y(y), idx(idx) {}

  mat x;
  double y;
  unsigned idx;
};

#endif
