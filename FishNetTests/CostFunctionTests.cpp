#include "stdafx.h"
#include "CppUnitTest.h"
#include "CostFunction.h"
#include "Tensor.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CostFunctionTests
{
  TEST_CLASS(CostFunctionTests)
  {
  public:
	TEST_METHOD(CrossEntropyZeroCost)
	{
	  ::CrossEntropyCostFunction cf;
	  Tensor activations({-1.0, -0.5, 0.0, 0.5, 1.0});
	  Tensor targets({-1.0, -0.5, 0.0, 0.5, 1.0});
	  Assert::AreEqual(0.0, cf.TotalCost(activations, targets));
	}
	TEST_METHOD(CrossEntropyZeroDerivative)
	{
	  ::CrossEntropyCostFunction cf;
	  Tensor activations({-1.0, -0.5, 0.0, 0.5, 1.0});
	  Tensor targets({-1.0, -0.5, 0.0, 0.5, 1.0});
	  Tensor out(5);
	  cf.Derivatives(activations, targets, out);
	  Assert::AreEqual(0.0, out.Get(0));
	  Assert::AreEqual(0.0, out.Get(1));
	  Assert::AreEqual(0.0, out.Get(2));
	  Assert::AreEqual(0.0, out.Get(3));
	  Assert::AreEqual(0.0, out.Get(4));
	}
  };
}
