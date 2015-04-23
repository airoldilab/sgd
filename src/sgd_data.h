#ifndef IMPLICIT_DATA_H
#define IMPLICIT_DATA_H

#include <iostream>
#include "sgd_basedef.h"

using namespace arma;

struct Imp_DataPoint;
struct Imp_Dataset;
struct Imp_OnlineOutput;
struct Imp_Size;

typedef boost::function<mat (const mat&, const Imp_DataPoint&, double)> score_func_type;
typedef boost::function<mat (const mat&, const Imp_DataPoint&, double, unsigned, unsigned)> learning_rate_type;

struct Imp_DataPoint {
  Imp_DataPoint(): x(mat()), y(0) {}
  Imp_DataPoint(mat xin, double yin):x(xin), y(yin) {}
//@members
  mat x;
  double y;
};

struct Imp_Dataset
{
  Imp_Dataset():X(mat()), Y(mat()) {}
  Imp_Dataset(mat xin, mat yin):X(xin), Y(yin) {}
//@members
  mat X;
  mat Y;
//@methods
  mat covariance() const {
    return cov(X);
  }

  friend std::ostream& operator<<(std::ostream& os, const Imp_Dataset& dataset) {
    os << "  Dataset:\n" << "    X has " << dataset.X.n_cols << " features\n" <<
          "    Total of " << dataset.X.n_rows << " data points" << std::endl;
    return os;
  }
};

struct Imp_OnlineOutput{
  //Construct Imp_OnlineOutput compatible with
  //the shape of data
  Imp_OnlineOutput(const Imp_Dataset& data, const mat& init)
   :estimates(mat(data.X.n_cols, data.X.n_rows)), initial(init) {}

  Imp_OnlineOutput(){}
//@members
  mat estimates;
  mat initial;
//@methods
  mat last_estimate(){
    return estimates.col(estimates.n_cols-1);
  }
};

struct Imp_Size {
  Imp_Size():nsamples(0), p(0){}
  Imp_Size(unsigned nin, unsigned pin):nsamples(nin), p(pin) {}
  unsigned nsamples;
  unsigned p;
};

#endif
