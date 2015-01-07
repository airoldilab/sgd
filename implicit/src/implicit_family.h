#ifndef IMPLICIT_FAMILY_H
#define IMPLICIT_FAMILY_H

#include "implicit_basedef.h"

using namespace arma;

struct Imp_Gaussian;
struct Imp_Poisson;
struct Imp_Binomial;

// gaussian model family
struct Imp_Gaussian {
  static std::string family;
  
  static double variance(double u) {
    return 1.;
  }

  static double deviance(const mat& y, const mat& mu, const mat& wt) {
    return sum(vec(wt % ((y-mu) % (y-mu))));
  }
};

std::string Imp_Gaussian::family = "gaussian";

// poisson model family
struct Imp_Poisson {
  static std::string family;
  
  static double variance(double u) {
    return u;
  }

  static double deviance(const mat& y, const mat& mu, const mat& wt) {
    vec r = vec(mu % wt);
    for (unsigned i = 0; i < r.n_elem; ++i) {
      if (y(i) > 0.) {
        r(i) = wt(i) * (y(i) * log(y(i)/mu(i)) - (y(i) - mu(i)));
      }
    }
    return sum(2. * r);
  }
};

std::string Imp_Poisson::family = "poisson";

// binomial model family
struct Imp_Binomial {
  static std::string family;
  
  static double variance(double u) {
    return u * (1. - u);
  }

  // In R the dev.resids of Binomial family is not exposed.
  // Found one [here](http://pages.stat.wisc.edu/~st849-1/lectures/GLMDeviance.pdf)
  static double deviance(const mat& y, const mat& mu, const mat& wt) {
    vec r(y.n_elem);
    for (unsigned i = 0; i < r.n_elem; ++i) {
      r(i) = 2. * wt(i) * (y_log_y(y(i), mu(i)) + y_log_y(1.-y(i), 1.-mu(i)));
    }
    return sum(r);
  }
private:
  static double y_log_y(double y, double mu) {
    return (y) ? (y * log(y/mu)) : 0.;
  }
};

std::string Imp_Binomial::family = "binomial";

#endif