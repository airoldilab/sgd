#ifndef IMPLICIT_TRANSFER_H
#define IMPLICIT_TRANSFER_H

#include "implicit_basedef.h"

using namespace arma;

struct Imp_Identity_Transfer;
struct Imp_Exp_Transfer;
struct Imp_Logistic_Transfer;

// Identity transfer function
struct Imp_Identity_Transfer {
  static double transfer(double u) {
    return u;
  }

  static mat transfer(const mat& u) {
    return u;
  }

  static double first_derivative(double u) {
    return 1.;
  }

  static double second_derivative(double u) {
    return 0.;
  }

  static bool valideta(double eta){
    return true;
  }
};

// Exponentional transfer function
struct Imp_Exp_Transfer {
  static double transfer(double u) {
    return exp(u);
  }

  static mat transfer(const mat& u) {
    mat result = mat(u);
    for (unsigned i = 0; i < result.n_rows; ++i) {
      result(i, 0) = transfer(u(i, 0));
    }
    return result;
  }

  static double first_derivative(double u) {
    return exp(u);
  }

  static double second_derivative(double u) {
    return exp(u);
  }

  static bool valideta(double eta){
    return true;
  }
};

// Logistic transfer function
struct Imp_Logistic_Transfer {
  static double transfer(double u) {
    return sigmoid(u);
  }

  static mat transfer(const mat& u) {
    mat result = mat(u);
    for (unsigned i = 0; i < result.n_rows; ++i) {
      result(i, 0) = transfer(u(i, 0));
    }
    return result;
  }

  static double first_derivative(double u) {
    double sig = sigmoid(u);
    return sig * (1. - sig);
  }

  static double second_derivative(double u) {
    double sig = sigmoid(u);
    return 2*pow(sig, 3) - 3*pow(sig, 2) + 2*sig;
  }

  static bool valideta(double eta){
    return true;
  }

private:
  // sigmoid function
  static double sigmoid(double u) {
      return 1. / (1. + exp(-u));
  }
};

#endif