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
  Sgd_OnlineOutput(const Sgd_Dataset& data, const mat& init, unsigned s=100)
   :estimates(mat(data.X.n_cols, s)), initial(init), crt_estimate(init),
    n_iter(data.X.n_rows), iter(0), size(s), n_recorded(0), pos(Mat<unsigned>(1, s)) {
      for (unsigned i=0; i < size; ++i) {
        pos(0, i) = int(round(pow(10, i * log10(n_iter) / (size-1))));
      }
      if (pos(0, pos.n_cols-1) != n_iter) 
        pos(0, pos.n_cols-1) = n_iter;
      if (n_iter < size)
        Rcpp::Rcout << "Warning: Too few data points for plotting!" << std::endl;
    }

  Sgd_OnlineOutput(){}

//@members
  mat estimates;
  mat initial;
  mat crt_estimate;
  unsigned n_iter; // Total number of iterations
  unsigned iter; // Current iteration
  unsigned size; // Number of coefs to be recorded
  unsigned n_recorded; //Number of coefs that have been recorded
  Mat<unsigned> pos; //The iteration of recorded coefficients

//@methods
  mat last_estimate() const{
    return crt_estimate;
  }

  Sgd_OnlineOutput& operator=(const mat& theta_new){
    crt_estimate = theta_new;
    iter += 1;
    if (iter == pos[n_recorded]){
      estimates.col(n_recorded) = theta_new;
      n_recorded += 1; 
      while (n_recorded < size && pos[n_recorded-1] == pos[n_recorded]){
        estimates.col(n_recorded) = theta_new;
        n_recorded += 1;
      }
        
    }
    return *this;
  }
};

struct Sgd_Size {
  Sgd_Size():nsamples(0), p(0){}
  Sgd_Size(unsigned nin, unsigned pin):nsamples(nin), p(pin) {}
  unsigned nsamples;
  unsigned p;
};

#endif
