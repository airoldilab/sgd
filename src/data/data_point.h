#ifndef DATA_DATA_POINT_H
#define DATA_DATA_POINT_H

#include <iostream>
#include <vector>
#include "basedef.h"

struct data_point {
  /* Collection for an individual observation and its response. */
  data_point() : x(mat()), y(0) {}
  data_point(mat xin, double yin) : x(xin), y(yin) {}

  mat x;
  double y;
};

#endif
