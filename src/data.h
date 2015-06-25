#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <vector>
#include "basedef.h"

using namespace arma;

struct Sgd_DataPoint;
struct Sgd_Dataset;
struct Sgd_OnlineOutput;

typedef boost::function<mat(const mat&, const Sgd_DataPoint&, double)> grad_func_type;
typedef boost::function<mat(const mat&, const Sgd_DataPoint&, double, unsigned, unsigned)> learning_rate_type;

struct Sgd_DataPoint {
  /* Collection for an individual observation and its response. */
  Sgd_DataPoint() : x(mat()), y(0) {}
  Sgd_DataPoint(mat xin, double yin) : x(xin), y(yin) {}

//@members
  mat x;
  double y;
};

struct Sgd_Dataset {
  /* Collection of all data points. */
  Sgd_Dataset(SEXP ptr, unsigned a, const boost::timer t) :
    X(mat()), Y(mat()), xpMat(ptr), t(t) {}

//@members
  mat X;
  mat Y;
  std::vector<unsigned> idxmap;
  unsigned n_samples;
  unsigned n_features;
  bool big;
  Rcpp::XPtr<BigMatrix> xpMat;
  boost::timer t;

//@methods
  void init(unsigned n_passes) {
    // Initialize number of columns and samples.
    unsigned nrow;
    if (!big) {
      nrow = X.n_rows;
      n_features = X.n_cols;
    } else {
      nrow = xpMat->nrow();
      n_features = xpMat->ncol();
    }
    n_samples = nrow * n_passes;
    // Initialize index mapping.
    idxmap = std::vector<unsigned>(n_samples);
    // std::srand(unsigned(std::time(0)));
    std::srand(0);
    for (unsigned i=0; i < n_passes; ++i) {
      for (unsigned j=0; j < nrow; ++j) {
        idxmap[i * nrow + j] = j;
      }
      std::random_shuffle(idxmap.begin() + i * nrow,
                          idxmap.begin() + (i + 1) * nrow);
    }
  }

  Sgd_DataPoint get_datapoint(unsigned t) const {
    /* Return the @t th data point */
    t = t - 1;
    mat xt;
    if (!big) {
      xt = mat(X.row(idxmap[t]));
    } else {
      MatrixAccessor<double> matacess(*xpMat);
      xt = mat(1, n_features);
      for (unsigned i=0; i < n_features; ++i) {
        xt(0, i) = matacess[i][idxmap[t]];
      }
    }
    double yt = Y(idxmap[t]);
    return Sgd_DataPoint(xt, yt);
  }

  mat covariance() const {
    return cov(X);
  }

  friend std::ostream& operator<<(std::ostream& os, const Sgd_Dataset& dataset) {
    os << "  Dataset:\n"
       << "    X has " << dataset.n_features << " features\n"
       << "    Total of " << dataset.n_samples << " data points" << std::endl;
    return os;
  }
};

struct Sgd_OnlineOutput {
  /* Collection of SGD-related values for the data set. */
  Sgd_OnlineOutput(const Sgd_Dataset& data, const mat& init, unsigned s=100) :
    estimates(mat(data.n_features, s)), initial(init), last_estimate(init),
    times(s), t(data.t), n_iter(data.n_samples), iter(0), size(s),
    n_recorded(0), pos(Mat<unsigned>(1, s)) {
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
  vec times;
  boost::timer t;
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
      times.at(n_recorded) = t.elapsed();
      n_recorded += 1;
      while (n_recorded < size && pos[n_recorded-1] == pos[n_recorded]) {
        estimates.col(n_recorded) = theta_new;
        times.at(n_recorded) = times.at(n_recorded-1);
        n_recorded += 1;
      }
    }
    return *this;
  }
};

#endif
