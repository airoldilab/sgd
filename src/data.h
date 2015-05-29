#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <vector>
#include "basedef.h"


using namespace arma;

struct Sgd_DataPoint;
struct Sgd_Dataset;
struct Sgd_OnlineOutput;
struct Sgd_Size;

typedef boost::function<mat (const mat&, const Sgd_DataPoint&, double)> grad_func_type;
typedef boost::function<mat (const mat&, const Sgd_DataPoint&, double, unsigned, unsigned)> learning_rate_type;

struct Sgd_DataPoint {
  Sgd_DataPoint(): x(mat()), y(0){}
  Sgd_DataPoint(mat xin, double yin):x(xin), y(yin){}
//@members
  mat x;
  double y;
};

struct Sgd_Dataset
{
  Sgd_Dataset(SEXP ptr, unsigned a):X(mat()), Y(mat()),xpMat(ptr)  {}
  // Sgd_Dataset(mat xin, mat yin):X(xin), Y(yin), xpMat(nullptr)  {}

//@members
  mat X;
  mat Y;
  std::vector<unsigned> idxmap;
  unsigned n_samples;
  unsigned n_cols;
  bool big;
  Rcpp::XPtr<BigMatrix> xpMat;


//@methods
  void init(unsigned n_passes) {
    unsigned nrow;
    if (!big){
      nrow = X.n_rows;
      n_cols = X.n_cols;
    } else{
      nrow = xpMat->nrow();
      n_cols = xpMat->ncol();
    }
    n_samples = nrow * n_passes;   
    idxmap = std::vector<unsigned>(n_samples);
    // std::srand(unsigned(std::time(0)));
    std::srand(0);
    for (unsigned i =0; i < n_passes; ++i) {
        for (unsigned j =0; j < nrow; ++j){
            idxmap[i * nrow + j] = j;
        }
        std::random_shuffle(idxmap.begin()+ i * nrow, idxmap.begin() + (i + 1) * nrow);
    }
  }
  mat covariance() const {
    return cov(X);
  }

  friend std::ostream& operator<<(std::ostream& os, const Sgd_Dataset& dataset) {
    os << "  Dataset:\n" << "    X has " << dataset.n_cols << " features\n" <<
          "    Total of " << dataset.n_samples << " data points" << std::endl;
    return os;
  }
};

struct Sgd_OnlineOutput
{
  //Construct Sgd_OnlineOutput compatible with
  //the shape of data
  Sgd_OnlineOutput(const Sgd_Dataset& data, const mat& init, unsigned s=100)
   : estimates(mat(data.n_cols, s)), initial(init), last_estimate(init),
    n_iter(data.n_samples), iter(0), size(s), n_recorded(0), pos(Mat<unsigned>(1, s)) {
      for (unsigned i=0; i < size; ++i) {
        pos(0, i) = int(round(pow(10, i * log10(n_iter) / (size-1))));
      }
      if (pos(0, pos.n_cols-1) != n_iter)
        pos(0, pos.n_cols-1) = n_iter;
      if (n_iter < size)
        Rcpp::Rcout << "Warning: Too few data points for plotting!" << std::endl;
    }

  Sgd_OnlineOutput() {}

//@members
  mat estimates;
  mat initial;
  mat last_estimate;
  unsigned n_iter; // Total number of iterations
  unsigned iter; // Current iteration
  unsigned size; // Number of coefs to be recorded
  unsigned n_recorded; //Number of coefs that have been recorded
  Mat<unsigned> pos; //The iteration of recorded coefficients

//@methods
  mat get_last_estimate() const {
    return last_estimate;
  }

  Sgd_OnlineOutput& operator=(const mat& theta_new) {
    last_estimate = theta_new;
    iter += 1;
    if (iter == pos[n_recorded]) {
      estimates.col(n_recorded) = theta_new;
      n_recorded += 1;
      while (n_recorded < size && pos[n_recorded-1] == pos[n_recorded]) {
        estimates.col(n_recorded) = theta_new;
        n_recorded += 1;
      }

    }
    return *this;
  }
};

struct Sgd_Size {
  Sgd_Size():nsamples(0), d(0) {}
  Sgd_Size(unsigned nin, unsigned din):nsamples(nin), d(din) {}
  unsigned nsamples;
  unsigned d;
};

#endif
