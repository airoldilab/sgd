#ifndef DATA_H
#define DATA_H

#include <iostream>
#include "basedef.h"

using namespace arma;

struct Sgd_DataPoint;
struct Sgd_Dataset;
struct Sgd_OnlineOutput;
struct Sgd_Size;

typedef boost::function<mat (const mat&, const Sgd_DataPoint&, double)> score_func_type;
typedef boost::function<mat (const mat&, const Sgd_DataPoint&, double, unsigned, unsigned)> learning_rate_type;

struct Sgd_DataPoint {
  Sgd_DataPoint(): x(mat()), y(0) {}
  Sgd_DataPoint(mat xin, double yin):x(xin), y(yin) {}
//@members
  mat x;
  double y;
};

struct Sgd_Dataset
{
  Sgd_Dataset():X(mat()), Y(mat()) {}
  Sgd_Dataset(mat xin, mat yin):X(xin), Y(yin) {}
//@members
  mat X;
  mat Y;
//@methods
  mat covariance() const {
    return cov(X);
  }

  friend std::ostream& operator<<(std::ostream& os, const Sgd_Dataset& dataset) {
    os << "  Dataset:\n" << "    X has " << dataset.X.n_cols << " features\n" <<
          "    Total of " << dataset.X.n_rows << " data points" << std::endl;
    return os;
  }
};

struct Sgd_OnlineOutput{
  //Construct Sgd_OnlineOutput compatible with
  //the shape of data
  Sgd_OnlineOutput(const Sgd_Dataset& data, const mat& init)
   :estimates(mat(data.X.n_cols, data.X.n_rows)), initial(init) {}

  Sgd_OnlineOutput(){}
//@members
  mat estimates;
  mat initial;
//@methods
  mat last_estimate(){
    return estimates.col(estimates.n_cols-1);
  }
};

struct Sgd_Size {
  Sgd_Size():nsamples(0), p(0){}
  Sgd_Size(unsigned nin, unsigned pin):nsamples(nin), p(pin) {}
  unsigned nsamples;
  unsigned p;
};

#endif
