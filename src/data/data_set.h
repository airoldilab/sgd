#ifndef DATA_DATA_SET_H
#define DATA_DATA_SET_H

#include "../basedef.h"
#include "data_point.h"

// wrapper around R's RNG such that we get a uniform distribution over
// [0,n) as required by the STL algorithm
inline int randWrapper(const int n) { return floor(unif_rand()*n); }

class data_set {
  /**
   * Collection of all data points.
   *
   * @param xpMat    pointer to bigmat if using bigmatrix
   * @param Xx       design matrix if not using bigmatrix
   * @param Yy       response values
   * @param n_passes number of passes for data
   * @param big      whether using bigmatrix or not
   * @param shuffle  whether to shuffle data set or not
   */
public:
  data_set(const SEXP& xpMat, const mat& Xx, const mat& Yy, unsigned n_passes,
    bool big, bool shuffle) : Y(Yy), big(big), xpMat_(xpMat), shuffle_(shuffle) {
    if (!big) {
      X = Xx;
      n_samples = X.n_rows;
      n_features = X.n_cols;
    } else {
      n_samples = xpMat_->nrow();
      n_features = xpMat_->ncol();
    }
    if (shuffle_) {
      idxvec_ = std::vector<unsigned>(n_samples*n_passes);
      for (unsigned i = 0; i < n_passes; ++i) {
        for (unsigned j = 0; j < n_samples; ++j) {
          idxvec_[i * n_samples + j] = j;
        }
        std::random_shuffle(idxvec_.begin() + i * n_samples,
                            idxvec_.begin() + (i + 1) * n_samples,
                            randWrapper);
      }
    }
  }

  // Index to the @t th data point
  data_point get_data_point(unsigned t) const {
    t = idxmap_(t - 1);
    mat xt;
    if (!big) {
      xt = mat(X.row(t));
    } else {
      MatrixAccessor<double> matacess(*xpMat_);
      xt = mat(1, n_features);
      for (unsigned i=0; i < n_features; ++i) {
        xt(0, i) = matacess[i][t];
      }
    }
    double yt = Y(t);
    return data_point(xt, yt, t);
  }

  mat X;
  mat Y;
  bool big;
  unsigned n_samples;
  unsigned n_features;

private:
  // index to data point for each iteration
  unsigned idxmap_(unsigned t) const {
    if (shuffle_) {
      return(idxvec_[t]);
    } else {
      return(t % n_samples);
    }
  }

  Rcpp::XPtr<BigMatrix> xpMat_;
  std::vector<unsigned> idxvec_;
  bool shuffle_;
};

#endif
