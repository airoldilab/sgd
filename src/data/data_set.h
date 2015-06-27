#ifndef DATA_DATA_SET_H
#define DATA_DATA_SET_H

#include <iostream>
#include <vector>
#include "basedef.h"
#include "data/data_point.h"

class data_set {
  /* Collection of all data points. */
public:
  mat X;
  mat Y;
  std::vector<unsigned> idxmap;
  unsigned n_samples;
  unsigned n_features;
  bool big;
  Rcpp::XPtr<BigMatrix> xpMat;
  boost::timer t;

  data_set(SEXP ptr, unsigned a, const boost::timer t) :
    X(mat()), Y(mat()), xpMat(ptr), t(t) {}

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

  data_point get_data_point(unsigned t) const {
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
    return data_point(xt, yt);
  }

  mat covariance() const {
    return cov(X);
  }

  friend std::ostream& operator<<(std::ostream& os, const data_set& data) {
    os << "  Data set:\n"
       << "    X has " << data.n_features << " features\n"
       << "    Total of " << data.n_samples << " data points" << std::endl;
    return os;
  }
};

#endif
