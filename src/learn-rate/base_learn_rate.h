#ifndef LEARN_RATE_BASE_LEARN_RATE_H
#define LEARN_RATE_BASE_LEARN_RATE_H

#include "basedef.h"
#include "data/data_point.h"
#include "learn-rate/learn_rate_value.h"

class base_learn_rate {
  /**
   * Base class for learning rates
   */
public:
  // Constructors
  base_learn_rate() {}

  // Operators
  virtual const learn_rate_value& operator()(const mat& theta_old, const
    data_point& data_pt, unsigned t, unsigned d) = 0;
};

#endif
