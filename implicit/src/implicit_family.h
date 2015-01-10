#ifndef IMPLICIT_FAMILY_H
#define IMPLICIT_FAMILY_H

#include "implicit_basedef.h"

using namespace arma;

struct Imp_Family_Base;
struct Imp_Gaussian;
struct Imp_Poisson;
struct Imp_Binomial;
struct Imp_Gamma;

struct Imp_Family_Base
{
  virtual double bfunc_for_score(double h) const = 0;
  virtual double variance(double u) const = 0;
  virtual double deviance(const mat& y, const mat& mu, const mat& wt) const = 0;
};

// gaussian model family
struct Imp_Gaussian : public Imp_Family_Base {
  virtual double bfunc_for_score(double h) const {
    return 1.;
  }

  virtual double variance(double u) const {
    return 1.;
  }

  virtual double deviance(const mat& y, const mat& mu, const mat& wt) const {
    return sum(vec(wt % ((y-mu) % (y-mu))));
  }
};

// poisson model family
struct Imp_Poisson : public Imp_Family_Base {
  virtual double bfunc_for_score(double h) const {
    if (h) {
      return 1. / h;
    }
    Rcpp::Rcout << "Out of valid range in b func for Poisson." << std::endl;
    return 1.;
  }
  
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

// binomial model family
struct Imp_Binomial : public Imp_Family_Base
{
  virtual double bfunc_for_score(double h) const {
    if (h > 0. && h < 1.) {
      return (1./h + 1./(1.-h));
    }
    Rcpp::Rcout << "Out of valid range in b func for Binomial." << std::endl;
    return 1.;
  }
  
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

struct Imp_Gamma : public Imp_Family_Base
{
  virtual double bfunc_for_score(double h) const {
    if (h) {
      return 1. / (h * h);
    }
    Rcpp::Rcout << "Out of valid range in b func for Gamma." << std::endl;
    return 1.;
  }

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