#ifndef FAMILY_H
#define FAMILY_H

#include "basedef.h"

using namespace arma;

class Sgd_Family_Base;
class Sgd_Gaussian;
class Sgd_Poisson;
class Sgd_Binomial;
class Sgd_Gamma;

class Sgd_Family_Base {
  /* Base class from which all exponential family classes inherit from */
public:
#if DEBUG
  virtual ~Sgd_Family_Base() {
    Rcpp::Rcout << "Family object released" << std::endl;
  }
#else
  virtual ~Sgd_Family_Base() {}
#endif

  virtual double variance(double u) const = 0;
  virtual double deviance(const mat& y, const mat& mu, const mat& wt) const = 0;
};

class Sgd_Gaussian : public Sgd_Family_Base {
  // gaussian model family
public:
  virtual double variance(double u) const {
    return 1.;
  }

  virtual double deviance(const mat& y, const mat& mu, const mat& wt) const {
    return sum(vec(wt % ((y-mu) % (y-mu))));
  }
};

class Sgd_Poisson : public Sgd_Family_Base {
  // poisson model family
public:
  virtual double variance(double u) const {
    return u;
  }

  virtual double deviance(const mat& y, const mat& mu, const mat& wt) const {
    vec r = vec(mu % wt);
    for (unsigned i = 0; i < r.n_elem; ++i) {
      if (y(i) > 0.) {
        r(i) = wt(i) * (y(i) * log(y(i)/mu(i)) - (y(i) - mu(i)));
      }
    }
    return sum(2. * r);
  }
};

class Sgd_Binomial : public Sgd_Family_Base {
  // binomial model family
public:
  virtual double variance(double u) const {
    return u * (1. - u);
  }

  // In R the dev.resids of Binomial family is not exposed.
  // Found one [here](http://pages.stat.wisc.edu/~st849-1/lectures/GLMDeviance.pdf)
  virtual double deviance(const mat& y, const mat& mu, const mat& wt) const {
    vec r(y.n_elem);
    for (unsigned i = 0; i < r.n_elem; ++i) {
      r(i) = 2. * wt(i) * (y_log_y(y(i), mu(i)) + y_log_y(1.-y(i), 1.-mu(i)));
    }
    return sum(r);
  }

private:
  double y_log_y(double y, double mu) const {
    return (y) ? (y * log(y/mu)) : 0.;
  }
};

class Sgd_Gamma : public Sgd_Family_Base {
  // gamma model family
public:
  virtual double variance(double u) const {
    return pow(u, 2);
  }

  virtual double deviance(const mat& y, const mat& mu, const mat& wt) const {
    vec r(y.n_elem);
    for (unsigned i = 0; i < r.n_elem; ++i) {
      r(i) = -2. * wt(i) * (log(y(i) ? y(i)/mu(i) : 1.) - (y(i)-mu(i)) / mu(i));
    }
    return sum(r);
  }
};

#endif
