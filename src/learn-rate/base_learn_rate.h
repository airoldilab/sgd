#ifndef LEARN_RATE_BASE_LEARN_RATE_H
#define LEARN_RATE_BASE_LEARN_RATE_H

#include "basedef.h"
#include "learn-rate/learn_rate_value.h"

class base_learn_rate {
  /**
   * Base class for learning rates
   */
public:
  base_learn_rate() {}

  virtual const learn_rate_value& operator()(unsigned t, const mat& grad_t) = 0;
};

#endif
