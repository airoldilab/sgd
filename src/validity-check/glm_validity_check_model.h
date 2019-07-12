#ifndef VALIDITY_CHECK_GLM_VALIDITY_CHECK_MODEL_H
#define VALIDITY_CHECK_GLM_VALIDITY_CHECK_MODEL_H

#include "../basedef.h"
#include "../data/data_set.h"
#include "../model/glm_model.h"

bool validity_check_model(const data_set& data, const mat& theta, unsigned t,
  const glm_model& model) {
  // TODO should this really be checked at each iteration?
  return true;
}

#endif
