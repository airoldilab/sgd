#ifndef TRANSFER_H
#define TRANSFER_H

#include "basedef.h"

using namespace arma;

class Sgd_Transfer_Base;
class Sgd_Identity_Transfer;
class Sgd_Inverse_Transfer;
class Sgd_Exp_Transfer;
class Sgd_Logistic_Transfer;

class Sgd_Transfer_Base {
  /* Base class from which all transfer function classes inherit from */
public:
#if DEBUG
  virtual ~Sgd_Transfer_Base() {
    Rcpp::Rcout << "Transfer object released! " << std::endl;
  }
#else
  virtual ~Sgd_Transfer_Base() {}
#endif

  virtual double transfer(double u) const = 0;

  virtual mat transfer(const mat& u) const {
    mat result = mat(u);
    for (unsigned i = 0; i < result.n_rows; ++i) {
      result(i, 0) = transfer(u(i, 0));
    }
    return result;
  }

  virtual double link(double u) const = 0;

  virtual double first_derivative(double u) const = 0;
  virtual double second_derivative(double u) const = 0;
  virtual bool valideta(double eta) const = 0;
};

class Sgd_Identity_Transfer : public Sgd_Transfer_Base {
  // Identity transfer function
public:
  virtual double transfer(double u) const {
    return u;
  }

  virtual double link(double u) const {
    return u;
  }

  virtual double first_derivative(double u) const {
    return 1.;
  }

  virtual double second_derivative(double u) const {
    return 0.;
  }

  virtual bool valideta(double eta) const {
    return true;
  }
};

class Sgd_Inverse_Transfer : public Sgd_Transfer_Base {
  // Inverse transfer function
public:
  virtual double transfer(double u) const {
    if (valideta(u)) {
      return -1. / u;
    }
    return 0.;
  }

  virtual double link(double u) const {
    if (u) {
      return -1. / u;
    }
    return 0.;
  }

  virtual double first_derivative(double u) const {
    if (valideta(u)) {
      return 1. / pow(u, 2);
    }
    return 0.;
  }

  virtual double second_derivative(double u) const {
    if (valideta(u)) {
      return -2. / pow(u, 3);
    }
    return 0.;
  }

  virtual bool valideta(double eta) const {
    return eta != 0;
  }
};

class Sgd_Exp_Transfer : public Sgd_Transfer_Base {
  // Exponentional transfer function
public:
  virtual double transfer(double u) const {
    return exp(u);
  }

  virtual double link(double u) const {
    if (u > 0.) {
      return log(u);
    }
    return 0.;
  }

  virtual double first_derivative(double u) const {
    return exp(u);
  }

  virtual double second_derivative(double u) const {
    return exp(u);
  }

  virtual bool valideta(double eta) const {
    return true;
  }
};

class Sgd_Logistic_Transfer : public Sgd_Transfer_Base {
  // Logistic transfer function
public:
  virtual double transfer(double u) const {
    return sigmoid(u);
  }

  virtual double link(double u) const {
    if (u > 0. && u < 1.) {
      return log(u / (1. - u));
    }
    return 0.;
  }

  virtual double first_derivative(double u) const {
    double sig = sigmoid(u);
    return sig * (1. - sig);
  }

  virtual double second_derivative(double u) const {
    double sig = sigmoid(u);
    return 2*pow(sig, 3) - 3*pow(sig, 2) + 2*sig;
  }

  virtual bool valideta(double eta) const {
    return true;
  }

private:
  double sigmoid(double u) const {
      return 1. / (1. + exp(-u));
  }
};

#endif
