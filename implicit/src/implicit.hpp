#include "RcppArmadillo.h"
#include <boost/math/tools/roots.hpp>
#include <math.h>

using namespace arma;

struct Imp_DataPoint;
struct Imp_Dataset;
struct Imp_OnlineOutput;
struct Imp_Identity;
struct Imp_Exp;
template<typename TRANSFER>
  struct Imp_Experiment;
struct Imp_Size;
struct Imp_Learning_rate;
struct Imp_transfer;


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

//Identity transfer function
struct Imp_Identity{
  double operator() (double u) const{
    return u;
  }
  double first(double u) const{
    return 1;
  }
  double second(double u) const{
    return 0;
  }
};

//exponent transfer function
struct Imp_Exp{
  double operator() (double u) const{
    return exp(u);
  }
  double first(double u) const{
    return exp(u);
  }
  double second(double u) const{
    return exp(u);
  }
};

template<typename TRANSFER>
struct Imp_Experiment {
//@members
  unsigned p;
  unsigned n_iters;
  Imp_Learning_rate lr;
  std::string model_name;
  TRANSFER h_transfer;
//@methods
  double learning_rate(unsigned t) const{
    if (model_name == "poisson")
      return double(10)/3/t;
    else if (model_name == "normal") {
      return lr(t);
    }
    return 0;
  }

  mat score_function(const mat& theta_old, const Imp_DataPoint& datapoint) const{
    return ((datapoint.y - h_transfer(as_scalar(datapoint.x * theta_old)))*datapoint.x).t();
  }

  //YKuang
  double h_first_derivative(double u) const{
    return h_transfer.first(u);
  }
  //YKuang
  double h_second_derivative(double u) const{
    return h_transfer.second(u);
  }

private:
  double sigmoid(double u) const {
    return 1. / (1. + exp(-u));
  }
};

struct Imp_Size{
  Imp_Size():nsamples(0), p(0){}
  Imp_Size(unsigned nin, unsigned pin):nsamples(nin), p(pin) {}
  unsigned nsamples;
  unsigned p;
};

// Compute score function coeff and its derivative for Implicit-SGD update
template<typename TRANSFER>
struct Get_score_coeff{
  Get_score_coeff(const Imp_Experiment<TRANSFER>& e, const Imp_DataPoint& d,
      const mat& t, double n):experiment(e), datapoint(d),
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
  const Imp_Experiment<TRANSFER>& experiment;
  const Imp_DataPoint& datapoint;
  const mat& theta_old;
  double normx;
};

// Root finding functor for Implicit-SGD update
template<typename TRANSFER>
struct Implicit_fn{
  typedef boost::math::tuple<double, double, double> tuple_type;
  Implicit_fn(double a, const Get_score_coeff& get_score): at(a), g(get_score){}
  tuple_type operator() (double u) const{
    double value = u - at * g(u);
    double first = 1 + at * g.first_derivative(u);
    double second = at * g.second_derivative(u);
    tuple_type result(value, first, second);
    return result;
  }
  double at;
  const Get_score_coeff<TRANSFER>& g;
};

