#include "RcppArmadillo.h"
#include <boost/math/tools/roots.hpp>
#include <boost/function.hpp>
#include <boost/bind/bind.hpp>
#include <boost/ref.hpp>
#include <math.h>
#include <string>
#include <cstddef>

using namespace arma;

#define nullptr NULL
#define DEBUG 1

struct Imp_DataPoint;
struct Imp_Dataset;
struct Imp_OnlineOutput;
struct Imp_Identity;
struct Imp_Exp;
struct Imp_Experiment;
struct Imp_Size;
struct Imp_Learning_rate;
struct Imp_transfer;

typedef boost::function<double (double)> uni_func_type;
typedef boost::function<mat (const Imp_DataPoint&, unsigned, unsigned)> learning_rate_type;

double sigmoid(double u);
double identity_transfer(double u);
double identity_first_deriv(double u);
double identity_second_deriv(double u);
double exp_transfer(double u);
double exp_first_deriv(double u);
double exp_second_deriv(double u);
double logistic_transfer(double u);
double logistic_first_deriv(double u);
double logistic_second_deriv(double u);

mat uni_dim_learning_rate(const Imp_DataPoint& data_pt, unsigned t, unsigned p,
                          double gamma, double alpha, double c, double scale);

struct Imp_DataPoint {
  Imp_DataPoint(): x(mat()), y(0) {}
  Imp_DataPoint(mat xin, double yin):x(xin), y(yin) {}
//@members
  mat x;
  double y;
};

struct Imp_Dataset
{
  Imp_Dataset():X(mat()), Y(mat()) {}
  Imp_Dataset(mat xin, mat yin):X(xin), Y(yin) {}
//@members
  mat X;
  mat Y;
//@methods
  mat covariance() const {
    return cov(X);
  }
};

struct Imp_OnlineOutput{
  //Construct Imp_OnlineOutput compatible with
  //the shape of data
  Imp_OnlineOutput(const Imp_Dataset& data):estimates(mat(data.X.n_cols, data.X.n_rows)){}
  Imp_OnlineOutput(){}
//@members
  mat estimates;
//@methods
  mat last_estimate(){
    return estimates.col(estimates.n_cols-1);
  }
};

struct Imp_Learning_rate {
  Imp_Learning_rate():gamma(1), alpha(1), c(1), scale(1) {}
  Imp_Learning_rate(double g, double a, double cin, double s):
    gamma(g), alpha(a), c(cin), scale(s) {}
  double gamma;
  double alpha;
  double c;
  double scale;
  double operator() (unsigned t) const {
    return scale * gamma * pow(1 + alpha * gamma * t, -c);
  }
};

mat uni_dim_learning_rate(const Imp_DataPoint& data_pt, unsigned t, unsigned p,
                          double gamma, double alpha, double c, double scale) {
  double lr = scale * gamma * pow(1 + alpha * gamma * t, -c);
  mat lr_mat = mat(p, p, fill::eye) * lr;
  return lr_mat;
}

double sigmoid(double u) {
    return 1. / (1. + exp(-u));
}

//Identity transfer function
double identity_transfer(double u) {
  return u;
}

double identity_first_deriv(double u) {
  return 1.;
}

double identity_second_deriv(double u) {
  return 0.;
}

//exponent transfer function
double exp_transfer(double u) {
  return exp(u);
}

double exp_first_deriv(double u) {
  return exp(u);
}

double exp_second_deriv(double u) {
  return exp(u);
}

//logistic transfer function
double logistic_transfer(double u) {
  return sigmoid(u);
}

double logistic_first_deriv(double u) {
  double sig = sigmoid(u);
  return sig * (1. - sig);
}

double logistic_second_deriv(double u) {
  double sig = sigmoid(u);
  return 2*pow(sig, 3) - 3*pow(sig, 2) + 2*sig;
}


struct Imp_Experiment {
//@members
  unsigned p;
  unsigned n_iters;
  //Imp_Learning_rate lr;
  std::string model_name;
//@methods
  Imp_Experiment(std::string transfer_name) {
    if (transfer_name == "identity") {
      //h_transfer_ = new Imp_Identity;
      transfer_ = &identity_transfer;
      transfer_first_deriv_ = &identity_first_deriv;
      transfer_second_deriv_ = &identity_second_deriv;
    }
    else if (transfer_name == "exp") {
      //h_transfer_ = new Imp_Exp;
      transfer_ = &exp_transfer;
      transfer_first_deriv_ = &exp_first_deriv;
      transfer_second_deriv_ = &exp_second_deriv;
    }
    else if (transfer_name == "logistic") {
      //h_transfer_ = new Imp_Logistic;
      transfer_ = &logistic_transfer;
      transfer_first_deriv_ = &logistic_first_deriv;
      transfer_second_deriv_ = &logistic_second_deriv;
    }
  }

  void init_uni_dim_learning_rate(double gamma, double alpha, double c, double scale) {
    lr_ = boost::bind(&uni_dim_learning_rate, _1, _2, _3, gamma, alpha, c, scale);
  }

  mat learning_rate(const Imp_DataPoint& data_pt, unsigned t) const{
    //return lr(t);
    return lr_(data_pt, t, p);
  }

  mat score_function(const mat& theta_old, const Imp_DataPoint& datapoint) const{
    return ((datapoint.y - h_transfer(as_scalar(datapoint.x * theta_old)))*datapoint.x).t();
  }

  double h_transfer(double u) const {
    return transfer_(u);
  }

  //YKuang
  double h_first_derivative(double u) const{
    return transfer_first_deriv_(u);
  }
  //YKuang
  double h_second_derivative(double u) const{
    return transfer_second_deriv_(u);
  }

private:
  uni_func_type transfer_;
  uni_func_type transfer_first_deriv_;
  uni_func_type transfer_second_deriv_;

  learning_rate_type lr_;
};

struct Imp_Size{
  Imp_Size():nsamples(0), p(0){}
  Imp_Size(unsigned nin, unsigned pin):nsamples(nin), p(pin) {}
  unsigned nsamples;
  unsigned p;
};

// Compute score function coeff and its derivative for Implicit-SGD update
struct Get_score_coeff{

  //Get_score_coeff(const Imp_Experiment<TRANSFER>& e, const Imp_DataPoint& d,
  Get_score_coeff(const Imp_Experiment& e, const Imp_DataPoint& d,
      const mat& t, double n) : experiment(e), datapoint(d),
    theta_old(t), normx(n) {}

  double operator() (double ksi) const{
    return datapoint.y-experiment.h_transfer(dot(theta_old, datapoint.x)
                     + normx * ksi);
  }

  double first_derivative (double ksi) const{
    return experiment.h_first_derivative(dot(theta_old, datapoint.x)
           + normx * ksi)*normx;
  }

  double second_derivative (double ksi) const{
    return experiment.h_second_derivative(dot(theta_old, datapoint.x)
             + normx * ksi)*normx*normx;
  }

  //const Imp_Experiment<TRANSFER>& experiment;
  const Imp_Experiment& experiment;
  const Imp_DataPoint& datapoint;
  const mat& theta_old;
  double normx;
};

// Root finding functor for Implicit-SGD update
struct Implicit_fn{
  typedef boost::math::tuple<double, double, double> tuple_type;

  //Implicit_fn(double a, const Get_score_coeff<TRANSFER>& get_score): at(a), g(get_score){}
  Implicit_fn(double a, const Get_score_coeff& get_score): at(a), g(get_score){}
  tuple_type operator() (double u) const{
    double value = u - at * g(u);
    double first = 1 + at * g.first_derivative(u);
    double second = at * g.second_derivative(u);
    tuple_type result(value, first, second);
    return result;
  }
  
  double at;
  //const Get_score_coeff<TRANSFER>& g;
  const Get_score_coeff& g;
};

