#include "stdafx.h"
#include "CostFunction.h"
#include "Tensor.h"

double CrossEntropyCostFunction::TotalCost(const Tensor& outputActivations, const Tensor& targetActivations) const
{
  const double* end = outputActivations.Elements() + outputActivations.Size();
  const double* target = targetActivations.Elements();
  double totalCost = 0.0;
  for(const double* actual = outputActivations.Elements(); actual < end; ++actual)
  {
	if (*actual < 1.0 - 1e-7)
	{
	  totalCost -= *target * log(*actual);
	  totalCost -= (1.0 - *target) * log(1.0 - *actual);
	}
	++target;
  }
  return totalCost;
}

void CrossEntropyCostFunction::Derivatives(const Tensor& outputActivations, const Tensor& targetActivations, Tensor& result) const
{
  // Return the vector of partial derivatives for the output activations.
  outputActivations.ComponentWiseSubtract(targetActivations, result);
}
