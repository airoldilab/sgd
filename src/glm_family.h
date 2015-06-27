#ifndef GLM_FAMILY_H
#define GLM_FAMILY_H

#include "basedef.h"

class base_family;
class gaussian_family;
class poisson_family;
class binomial_family;
class gamma_family;

class base_family {
  /* Base class from which all exponential family classes inherit from */
public:
#if DEBUG
  virtual ~base_family() {
    Rcpp::Rcout << "Family object released" << std::endl;
  }
#else
  virtual ~base_family() {}
#endif

  virtual double variance(double u) const = 0;
  virtual double deviance(const mat& y, const mat& mu, const mat& wt) const = 0;
};

class gaussian_family : public base_family {
  // gaussian model family
public:
  virtual double variance(double u) const {
    return 1.;
  }

  virtual double deviance(const mat& y, const mat& mu, const mat& wt) const {
    return sum(vec(wt % ((y-mu) % (y-mu))));
  }
};

class poisson_family : public base_family {
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

class binomial_family : public base_family {
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

class gamma_family : public base_family {
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
