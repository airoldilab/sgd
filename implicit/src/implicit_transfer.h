#ifndef IMPLICIT_TRANSFER_H
#define IMPLICIT_TRANSFER_H

#include "implicit_basedef.h"

using namespace arma;

struct Imp_Transfer_Base;
struct Imp_Identity_Transfer;
struct Imp_Inverse_Transfer;
struct Imp_Exp_Transfer;
struct Imp_Logistic_Transfer;

struct Imp_Transfer_Base {
#if DEBUG
  virtual ~Imp_Transfer_Base() {
    Rcpp::Rcout << "Transfer object released! " << std::endl;
  }
#endif

  virtual double transfer(double u) const = 0;

  virtual mat transfer(const mat& u) const {
    mat result = mat(u);
    for (unsigned i = 0; i < result.n_rows; ++i) {
      result(i, 0) = transfer(u(i, 0));
    }
    return result;
  }

  virtual double first_derivative(double u) const = 0;
  virtual double second_derivative(double u) const = 0;
  virtual bool valideta(double eta) const = 0;
};

// Identity transfer function
struct Imp_Identity_Transfer : public Imp_Transfer_Base {
  virtual double transfer(double u) const {
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

// Inverse transfer function
struct Imp_Inverse_Transfer : public Imp_Transfer_Base {
  virtual double transfer(double u) const {
    if (valideta(u)) {
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

// Exponentional transfer function
struct Imp_Exp_Transfer : public Imp_Transfer_Base {
  virtual double transfer(double u) const {
    return exp(u);
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

// Logistic transfer function
struct Imp_Logistic_Transfer : public Imp_Transfer_Base {
  virtual double transfer(double u) const {
    return sigmoid(u);
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
  // sigmoid function
  double sigmoid(double u) const {
      return 1. / (1. + exp(-u));
  }
};

#endif