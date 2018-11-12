#include "stdafx.h"
#include "CppUnitTest.h"
#include "ActivationFunction.h"
#include "Tensor.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace ActivationFunctionTests
{
  TEST_CLASS(ActivationFunctionTests)
  {
  public:
	TEST_METHOD(TanH)
	{
	  ::TanH tanh;
	  Tensor tensor(std::initializer_list<double>{-1, -0.5, 0, 0.5, 1});
	  tanh.Apply(tensor);
	  Assert::AreEqual(-0.76159415, tensor.Get(0), 1e-6);
	  Assert::AreEqual(-0.46211716, tensor.Get(1), 1e-6);
	  Assert::AreEqual(0.0, tensor.Get(2), 1e-6);
	  Assert::AreEqual(0.46211716, tensor.Get(3), 1e-6);
	  Assert::AreEqual(0.76159415, tensor.Get(4), 1e-6);
	}
	TEST_METHOD(TanHDerivative)
	{
	  ::TanH tanh;
	  Tensor in(std::initializer_list<double>{-1, -0.5, 0, 0.5, 1});
	  Tensor out(5);
	  tanh.ApplyDerivative(in, out);
	  Assert::AreEqual(0.4199743, out.Get(0), 1e-6);
	  Assert::AreEqual(0.7864477, out.Get(1), 1e-6);
	  Assert::AreEqual(1.0, out.Get(2), 1e-6);
	  Assert::AreEqual(0.7864477, out.Get(3), 1e-6);
	  Assert::AreEqual(0.4199743, out.Get(4), 1e-6);
	}
  };
}
