#include "stdafx.h"
#include "CppUnitTest.h"
#include "Tensor.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace FishNetTests
{		
  TEST_CLASS(TensorTests)
  {
  public:
	TEST_METHOD(Construct3DTensorFromInitializerList)
	{
	  Tensor t1(std::initializer_list<double>{2, 3, 5, 7, 11, 13, 17, 19, 23, 50, 27, 27, 3, -14, 33, 17, 18, -46},
		2, 3, 3);
	  Assert::AreEqual<size_t>(18, t1.Size());
	  Assert::AreEqual<size_t>(1, t1.Hyperplanes());
	  Assert::AreEqual<uint32_t>(2, t1.Planes());
	  Assert::AreEqual<uint32_t>(3, t1.Rows());
	  Assert::AreEqual<uint32_t>(3, t1.Columns());
	  Assert::AreEqual<double>(2, t1.Get(0, 0, 0));
	  Assert::AreEqual<double>(3, t1.Get(0, 0, 1));
	  Assert::AreEqual<double>(5, t1.Get(0, 0, 2));
	  Assert::AreEqual<double>(7, t1.Get(0, 1, 0));
	  Assert::AreEqual<double>(11, t1.Get(0, 1, 1));
	  Assert::AreEqual<double>(13, t1.Get(0, 1, 2));
	  Assert::AreEqual<double>(17, t1.Get(0, 2, 0));
	  Assert::AreEqual<double>(19, t1.Get(0, 2, 1));
	  Assert::AreEqual<double>(23, t1.Get(0, 2, 2));
	  Assert::AreEqual<double>(50, t1.Get(1, 0, 0));
	  Assert::AreEqual<double>(27, t1.Get(1, 0, 1));
	  Assert::AreEqual<double>(27, t1.Get(1, 0, 2));
	  Assert::AreEqual<double>(3, t1.Get(1, 1, 0));
	  Assert::AreEqual<double>(-14, t1.Get(1, 1, 1));
	  Assert::AreEqual<double>(33, t1.Get(1, 1, 2));
	  Assert::AreEqual<double>(17, t1.Get(1, 2, 0));
	  Assert::AreEqual<double>(18, t1.Get(1, 2, 1));
	  Assert::AreEqual<double>(-46, t1.Get(1, 2, 2));
	}

	TEST_METHOD(GetAndSetIn2DTensor)
	{
	  Tensor t(3, 4);
	  Assert::AreEqual<size_t>(1, t.Hyperplanes());
	  Assert::AreEqual<size_t>(1, t.Planes());
	  Assert::AreEqual<uint32_t>(3, t.Rows());
	  Assert::AreEqual<uint32_t>(4, t.Columns());

	  t.Set(0, 0, 1.227);
	  t.Set(0, 1, 6.127);
	  t.Set(0, 2, 5.638);
	  t.Set(1, 0, 1.072);
	  t.Set(1, 1, 5.029);
	  t.Set(1, 2, 1.142);
	  t.Set(3, 3, 0.074);
	  t.Set(2, 1, 7.063);
	  t.Set(2, 3, 2.512);

	  Assert::AreEqual<double>(t.Get(0, 0), 1.227);
	  Assert::AreEqual<double>(t.Get(0, 1), 6.127);
	  Assert::AreEqual<double>(t.Get(0, 2), 5.638);
	  Assert::AreEqual<double>(t.Get(1, 0), 1.072);
	  Assert::AreEqual<double>(t.Get(1, 1), 5.029);
	  Assert::AreEqual<double>(t.Get(1, 2), 1.142);
	  Assert::AreEqual<double>(t.Get(3, 3), 0.074);
	  Assert::AreEqual<double>(t.Get(2, 1), 7.063);
	  Assert::AreEqual<double>(t.Get(2, 3), 2.512);
	}

	TEST_METHOD(GetFrom4DTensor)
	{
	  std::array<double, 81> values =
	  {
		-0.23702, 0.21150, 0.65438, -0.42726, -0.24743, 0.17577, -0.69894, 0.88361, -0.40443,
		0.93639, 0.52495, 0.25173, 0.11524, -0.97168, -0.67457, -0.37762, 0.05999, -0.09300,
		-0.89895, -0.41392, 0.11044, -0.08912, -0.76731, 0.20898, 0.79850, -0.52233, 0.33790,
		0.62486, 0.48724, -0.41774, -0.61468, -0.61290, -0.27791, 0.19407, 0.70756, -0.15033,
		0.71936, -0.33090, -0.80612, -0.81910, -0.17306, -0.25753, 0.36237, 0.21850, -0.30818,
		-0.12927, -0.79222, -0.71353, 0.55857, 0.83025, 0.87724, -0.12399, -0.90516, 0.94674,
		0.43601, 0.06888, -0.38997, -0.55446, -0.39819, 0.79830, 0.73636, 0.87757, 0.72745,
		-0.72397, -0.76952, -0.40069, -0.02191, 0.16214, -0.29024, 0.94557, 0.23866, -0.39968,
		0.81240, -0.40659, 0.42175, -0.45065, 0.58492, 0.17362, -0.89579, 0.41289, 0.05642
	  };

	  Tensor t(std::initializer_list<double>(values.data(), values.data() + 81), 3, 3, 3, 3);
  	  Assert::AreEqual<size_t>(3, t.Hyperplanes());
	  Assert::AreEqual<size_t>(3, t.Planes());
	  Assert::AreEqual<uint32_t>(3, t.Rows());
	  Assert::AreEqual<uint32_t>(3, t.Columns());

	  int i = 0;
	  for (int dim4 = 0; dim4 < 3; ++dim4)
	  {
		for (int dim3 = 0; dim3 < 3; ++dim3)
		{
		  for (int row = 0; row < 3; ++row)
		  {
			for (int col = 0; col < 3; ++col)
			{
			  Assert::AreEqual<double>(t.Get(dim4, dim3, row, col), values[i++]);
			}
		  }
		}
	  }
	}

	TEST_METHOD(ElementAddressIn2DTensor)
	{
	  Tensor t1(9, 10);
	  Assert::AreEqual<size_t>(1, t1.Hyperplanes());
	  Assert::AreEqual<size_t>(1, t1.Planes());
	  Assert::AreEqual<uint32_t>(9, t1.Rows());
	  Assert::AreEqual<uint32_t>(10, t1.Columns());
	  double* start = t1.Elements();

	  Assert::AreEqual<double*>(start, t1.ElementAddress(0, 0));
	  Assert::AreEqual<size_t>(30, t1.ElementAddress(3, 0) - start);
	  Assert::AreEqual<size_t>(30 + 5, t1.ElementAddress(3, 5) - start);
	}

	TEST_METHOD(ElementAddressIn3DTensor)
	{
	  Tensor t1(8, 9, 10);
	  Assert::AreEqual<uint32_t>(1, t1.Hyperplanes());
	  Assert::AreEqual<uint32_t>(8, t1.Planes());
	  Assert::AreEqual<uint32_t>(9, t1.Rows());
	  Assert::AreEqual<uint32_t>(10, t1.Columns());
	  double* start = t1.Elements();

	  Assert::AreEqual<double*>(start, t1.ElementAddress(0, 0, 0));
	  Assert::AreEqual<double*>(start + 90, t1.ElementAddress(1, 0, 0));
	  Assert::AreEqual<double*>(start + 90 + 30, t1.ElementAddress(1, 3, 0));
	  Assert::AreEqual<double*>(start + 90 + 30 + 5, t1.ElementAddress(1, 3, 5));
	}

	TEST_METHOD(ElementAddressIn4DTensor)
	{
	  Tensor t1(7, 8, 9, 10);
	  Assert::AreEqual<uint32_t>(7, t1.Hyperplanes());
	  Assert::AreEqual<uint32_t>(8, t1.Planes());
	  Assert::AreEqual<uint32_t>(9, t1.Rows());
	  Assert::AreEqual<uint32_t>(10, t1.Columns());
	  double* start = t1.Elements();

	  Assert::AreEqual<double*>(start, t1.ElementAddress(0, 0, 0, 0));
	  Assert::AreEqual<double*>(start + 90, t1.ElementAddress(0, 1, 0, 0));
	  Assert::AreEqual<double*>(start + 90 + 30, t1.ElementAddress(0, 1, 3, 0));
	  Assert::AreEqual<double*>(start + 90 + 30 + 5, t1.ElementAddress(0, 1, 3, 5));
	  Assert::AreEqual<double*>(start + (5 * 8 * 90) + 90 + 30 + 5, t1.ElementAddress(5, 1, 3, 5));
	}


	TEST_METHOD(ComponentwiseAdd1)
	{
	  Tensor t1(std::initializer_list<double>{ 2, 3, 5, 7, 11 }, 1, 5);
	  Tensor t2(std::initializer_list<double>{ 4, 7, 1, 0, 33 }, 1, 5);
	  t1.ComponentWiseAdd(t2);
  	  Assert::AreEqual<size_t>(1, t1.Hyperplanes());
	  Assert::AreEqual<size_t>(1, t1.Planes());
	  Assert::AreEqual<uint32_t>(1, t1.Rows());
	  Assert::AreEqual<uint32_t>(5, t1.Columns());
	  Assert::AreEqual<double>(6, t1.Get(0, 0));
	  Assert::AreEqual<double>(10, t1.Get(0, 1));
	  Assert::AreEqual<double>(6, t1.Get(0, 2));
	  Assert::AreEqual<double>(7, t1.Get(0, 3));
	  Assert::AreEqual<double>(44, t1.Get(0, 4));
	}

	TEST_METHOD(ComponentwiseAdd2)
	{
	  Tensor t1(std::initializer_list<double>{ 2, 3, 5, 7, 11 }, 1, 5);
	  Tensor t2(std::initializer_list<double>{ 4, 7, 1, 0, 33 }, 1, 5);
	  Tensor result(5);
	  t1.ComponentWiseAdd(t2, result);
	  Assert::AreEqual<uint32_t>(5, result.Size());
	  Assert::AreEqual<double>(6, result.Get(0));
	  Assert::AreEqual<double>(10, result.Get(1));
	  Assert::AreEqual<double>(6, result.Get(2));
	  Assert::AreEqual<double>(7, result.Get(3));
	  Assert::AreEqual<double>(44, result.Get(4));
	}

	TEST_METHOD(ComponentwiseSubtract1)
	{
	  Tensor t1(std::initializer_list<double>{ 2, 3, 5, 7, 11 }, 1, 5);
	  Tensor t2(std::initializer_list<double>{ 4, 7, 1, 0, 33 }, 1, 5);
	  t1.ComponentWiseSubtract(t2);
  	  Assert::AreEqual<size_t>(1, t1.Hyperplanes());
	  Assert::AreEqual<size_t>(1, t1.Planes());
	  Assert::AreEqual<uint32_t>(1, t1.Rows());
	  Assert::AreEqual<uint32_t>(5, t1.Columns());
	  Assert::AreEqual<double>(-2, t1.Get(0, 0));
	  Assert::AreEqual<double>(-4, t1.Get(0, 1));
	  Assert::AreEqual<double>(4, t1.Get(0, 2));
	  Assert::AreEqual<double>(7, t1.Get(0, 3));
	  Assert::AreEqual<double>(-22, t1.Get(0, 4));
	}

	TEST_METHOD(ComponentwiseSubtract2)
	{
	  Tensor t1(std::initializer_list<double>{ 2, 3, 5, 7, 11 }, 1, 5);
	  Tensor t2(std::initializer_list<double>{ 4, 7, 1, 0, 33 }, 1, 5);
	  Tensor result(5);
	  t1.ComponentWiseSubtract(t2, result);
	  Assert::AreEqual<uint32_t>(5, result.Size());
	  Assert::AreEqual<double>(-2, result.Get(0));
	  Assert::AreEqual<double>(-4, result.Get(1));
	  Assert::AreEqual<double>(4, result.Get(2));
	  Assert::AreEqual<double>(7, result.Get(3));
	  Assert::AreEqual<double>(-22, result.Get(4));
	}

	TEST_METHOD(ComponentwiseMultiply1)
	{
	  Tensor t1(std::initializer_list<double>{ 2, 3, 5, 7, 11 }, 1, 5);
	  Tensor t2(std::initializer_list<double>{ 4, 7, 1, 0, 33 }, 1, 5);
	  t1.ComponentWiseMultiply(t2);
	  Assert::AreEqual<uint32_t>(1, t1.Hyperplanes());
	  Assert::AreEqual<uint32_t>(1, t1.Planes());
	  Assert::AreEqual<uint32_t>(1, t1.Rows());
	  Assert::AreEqual<uint32_t>(5, t1.Columns());
	  Assert::AreEqual<double>(8, t1.Get(0, 0));
	  Assert::AreEqual<double>(21, t1.Get(0, 1));
	  Assert::AreEqual<double>(5, t1.Get(0, 2));
	  Assert::AreEqual<double>(0, t1.Get(0, 3));
	  Assert::AreEqual<double>(363, t1.Get(0, 4));
	}

	TEST_METHOD(ComponentwiseMultiply2)
	{
	  Tensor t1(std::initializer_list<double>{ 2, 3, 5, 7, 11 }, 1, 5);
	  Tensor t2(std::initializer_list<double>{ 4, 7, 1, 0, 33 }, 1, 5);
	  Tensor result(5);
	  t1.ComponentWiseMultiply(t2, result);
	  Assert::AreEqual<uint32_t>(5, result.Size());
	  Assert::AreEqual<double>(8, result.Get(0));
	  Assert::AreEqual<double>(21, result.Get(1));
	  Assert::AreEqual<double>(5, result.Get(2));
	  Assert::AreEqual<double>(0, result.Get(3));
	  Assert::AreEqual<double>(363, result.Get(4));
	}
  };
}