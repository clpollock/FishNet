#include "stdafx.h"
#include "CppUnitTest.h"
#include "DropoutMask.h"
#include "Layer.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace FullyConnectedLayerTests
{
  TEST_CLASS(FullyConnectedLayerTests)
  {
  public:
	TEST_METHOD(FullyConnectedLayerFeedForward)
	{
	  auto weights = std::make_unique<Tensor>(std::initializer_list<double>{
		0.838504, 0.422149, 0.288635, 0.907155,
		  0.792704, 0.847105, 0.265283, 0.122859,
		  0.184963, 0.261111, 0.743236, 0.174590
	  }, 3, 4);
	  auto biases = std::make_unique<Tensor>(std::initializer_list<double>{ -0.1, 0.5, 0.0 });
	  Tensor input(std::initializer_list<double>{ 2.0, 3.0, 4.0, 5.0 });
	  double expectedOutput[] = { 8.53377, 6.30215, 4.999153 };

	  FullyConnectedLayer layer(std::move(weights), std::move(biases), nullptr);
	  Tensor output(3);
	  layer.FeedForward(input, output, nullptr);

	  Assert::AreEqual(expectedOutput[0], output.Get(0), 1e-5);
	  Assert::AreEqual(expectedOutput[1], output.Get(1), 1e-5);
	  Assert::AreEqual(expectedOutput[2], output.Get(2), 1e-5);
	}

	TEST_METHOD(FullyConnectedLayerFeedForwardWithDropout)
	{
	  auto weights = std::make_unique<Tensor>(std::initializer_list<double>{
		0.838504, 0.422149, 0.288635, 0.907155,
		  0.792704, 0.847105, 0.265283, 0.122859,
		  0.184963, 0.261111, 0.743236, 0.174590
	  }, 3, 4);
	  auto biases = std::make_unique<Tensor>(std::initializer_list<double>{ -0.1, 0.5, 0.0 });
	  Tensor input(std::initializer_list<double>{ 2.0, 3.0, 4.0, 5.0 });

	  std::array<DropoutMask, 8> masks =
	  {
		DropoutMask({ true, true, true }),
		DropoutMask({ false, false, false }),
		DropoutMask({ false, false, true }),
		DropoutMask({ false, true, false }),
		DropoutMask({ false, true, true }),
		DropoutMask({ true, false, false }),
		DropoutMask({ true, false, true }),
		DropoutMask({ true, true, false })
	  };

	  double expected[8][3] =
	  {
		{ 8.53377, 6.30215, 4.999153 },
		{ 0.0, 0.0, 0.0 },
		{ 0.0, 0.0, 4.999153 },
		{ 0.0, 6.30215, 0.0 },
		{ 0.0, 6.30215, 4.999153 },
		{ 8.53377, 0.0, 0.0 },
		{ 8.53377, 0.0, 4.999153 },
		{ 8.53377, 6.30215, 0.0 }
	  };

	  FullyConnectedLayer layer(std::move(weights), std::move(biases), nullptr);
	  Tensor output(3);
	  for (int i = 0; i < 8; ++i)
	  {
		layer.FeedForward(input, output, &masks[i]);
		for (int j = 0; j < 3; ++j)
		{
		  std::wostringstream msg;
		  msg << "Mismatch in mask " << i << ", entry " << j;
		  Assert::AreEqual(expected[i][j], output.Get(j), 1e-5, msg.str().c_str());
		}
	  }
	}

	TEST_METHOD(FullyConnectedLayerSwitchBetweenTestingAndTrainingWeights)
	{
	  std::initializer_list<double> weights = {
		0.838504, 0.422149, 0.288635, 0.907155,
		0.792704, 0.847105, 0.265283, 0.122859,
		0.184963, 0.261111, 0.743236, 0.174590
	  };
	  auto weightsTensor = std::make_unique<Tensor>(weights, 3, 4);
	  auto biaseTensor = std::make_unique<Tensor>(std::initializer_list<double>{ -0.1, 0.5, 0.0 });
	  FullyConnectedLayer layer(std::move(weightsTensor), std::move(biaseTensor), nullptr, 1.0, 0.6);
	  // Verify that weights are initially set as we expect.
	  const double* weight = weights.begin();
	  for (int32_t i = 0; i < 12; ++i)
	  {
		Assert::AreEqual(*weight, layer.Weights().Get(i), 1e-5);
		++weight;
	  }
	  // Test switching back to testing weights.
	  layer.SwitchToTestingWeights();
	  weight = weights.begin();
	  for (int32_t i = 0; i < 12; ++i)
	  {
		Assert::AreEqual(*weight * 0.6, layer.Weights().Get(i), 1e-5);
		++weight;
	  }
	  // Test switching back to training weights.
	  layer.SwitchToTrainingWeights();
	  weight = weights.begin();
	  for (int32_t i = 0; i < 12; ++i)
	  {
		Assert::AreEqual(*weight, layer.Weights().Get(i), 1e-5);
		++weight;
	  }
	}

	TEST_METHOD(FullyConnectedLayerBackpropagateError)
	{
	  auto weights = std::make_unique<Tensor>(std::initializer_list<double>{
		0.838504, 0.422149, 0.288635, 0.907155,
		  0.792704, 0.847105, 0.265283, 0.122859,
		  0.184963, 0.261111, 0.743236, 0.174590
	  }, 3, 4);
	  auto biases = std::make_unique<Tensor>(std::initializer_list<double>{ -0.1, 0.5, 0.0 });
	  FullyConnectedLayer layer(std::move(weights), std::move(biases), nullptr);
	  Tensor errorInThisLayer(std::initializer_list<double>{ 0.3, -0.015, 0.677 });
	  Tensor errorinPrevLayer(4);
	  layer.BackpropagateError(errorInThisLayer, errorinPrevLayer, nullptr);

	  for (int prevI = 0; prevI < 4; ++prevI)
	  {
		double expectedError = 0.0;
		for (int thisI = 0; thisI < 3; ++thisI)
		  expectedError += errorInThisLayer.Get(thisI) * layer.Weights().Get(thisI, prevI);
		Assert::AreEqual(expectedError, errorinPrevLayer.Get(prevI), 1e-5);
	  }
	}

	TEST_METHOD(FullyConnectedLayerBackpropagateErrorWithDropout)
	{
	  auto weights = std::make_unique<Tensor>(std::initializer_list<double>{
		0.838504, 0.422149, 0.288635, 0.907155,
		  0.792704, 0.847105, 0.265283, 0.122859,
		  0.184963, 0.261111, 0.743236, 0.174590
	  }, 3, 4);
	  auto biases = std::make_unique<Tensor>(std::initializer_list<double>{ -0.1, 0.5, 0.0 });
	  FullyConnectedLayer layer(std::move(weights), std::move(biases), nullptr);
	  Tensor errorInThisLayer(std::initializer_list<double>{ 0.3, -0.015, 0.677 });
	  Tensor errorinPrevLayer(4);

	  std::array<DropoutMask, 8> masks =
	  {
		DropoutMask({ true, true, true }),
		DropoutMask({ false, false, false }),
		DropoutMask({ false, false, true }),
		DropoutMask({ false, true, false }),
		DropoutMask({ false, true, true }),
		DropoutMask({ true, false, false }),
		DropoutMask({ true, false, true }),
		DropoutMask({ true, true, false })
	  };

	  for (int mi = 0; mi < 8; ++ mi)
	  {
		layer.BackpropagateError(errorInThisLayer, errorinPrevLayer, &masks[mi]);
		for (int prevI = 0; prevI < 4; ++prevI)
		{
		  double expectedError = 0.0;
		  const DropoutMask& mask = masks[mi];
		  for (int thisI = 0; thisI < 3; ++thisI)
		  {
			if (mask.Get(thisI))
			  expectedError += errorInThisLayer.Get(thisI) * layer.Weights().Get(thisI, prevI);
		  }
		  std::wostringstream msg;
		  msg << "Mismatch in mask " << mi << ", previous layer neuron " << prevI;
		  Assert::AreEqual(expectedError, errorinPrevLayer.Get(prevI), 1e-5, msg.str().c_str());
		}
	  }
	}

	TEST_METHOD(FullyConnectedLayerUpdateWeightAndBiasErrors)
	{
	  auto weights = std::make_unique<Tensor>(std::initializer_list<double>{
		0.838504, 0.422149, 0.288635, 0.907155,
		0.792704, 0.847105, 0.265283, 0.122859,
		0.184963, 0.261111, 0.743236, 0.174590
	  }, 3, 4);
	  auto biases = std::make_unique<Tensor>(std::initializer_list<double>{ -0.1, 0.5, 0.0 });
	  FullyConnectedLayer layer(std::move(weights), std::move(biases), nullptr);
	  Tensor delta(std::initializer_list<double>{ 0.092, 0.765, 0.624 });
	  Tensor previousLayerActivations(std::initializer_list<double>{ 0.32, 0.0635, 0.71, 0.10034 });
	  Tensor nablaW(3, 4);
	  Tensor nablaB(3);
	  layer.UpdateWeightAndBiasErrors(delta, previousLayerActivations, nablaW, nablaB, nullptr);

	  for (uint32_t i = 0; i < 3; ++i)
	  {
		Assert::AreEqual(delta.Get(i), nablaB.Get(i));
		for (uint32_t j = 0; j < 4; ++j)
		{
		  double expected = delta.Get(i) * previousLayerActivations.Get(j);
		  Assert::AreEqual(expected, nablaW.Get(i, j), 1e-5);
		}
	  }
	}

	TEST_METHOD(FullyConnectedLayerUpdateWeightAndBiasErrorsWithDropout)
	{
	  auto weights = std::make_unique<Tensor>(std::initializer_list<double>{
		0.838504, 0.422149, 0.288635, 0.907155,
		  0.792704, 0.847105, 0.265283, 0.122859,
		  0.184963, 0.261111, 0.743236, 0.174590
	  }, 3, 4);
	  auto biases = std::make_unique<Tensor>(std::initializer_list<double>{ -0.1, 0.5, 0.0 });
	  FullyConnectedLayer layer(std::move(weights), std::move(biases), nullptr);
	  Tensor delta(std::initializer_list<double>{ 0.092, 0.765, 0.624 });
	  Tensor previousLayerActivations(std::initializer_list<double>{ 0.32, 0.0635, 0.71, 0.10034 });

	  std::array<DropoutMask, 8> masks =
	  {
		DropoutMask({ true, true, true }),
		DropoutMask({ false, false, false }),
		DropoutMask({ false, false, true }),
		DropoutMask({ false, true, false }),
		DropoutMask({ false, true, true }),
		DropoutMask({ true, false, false }),
		DropoutMask({ true, false, true }),
		DropoutMask({ true, true, false })
	  };

	  for (int mi = 0; mi < 8; ++mi)
	  {
		Tensor nablaW(3, 4);
		Tensor nablaB(3);
		const DropoutMask& mask = masks[mi];
		layer.UpdateWeightAndBiasErrors(delta, previousLayerActivations, nablaW, nablaB, &mask);
		for (uint32_t i = 0; i < 3; ++i)
		{
		  if (mask.Get(i))
		  {
			std::wostringstream msg;
			msg << "Mismatch in mask " << mi << ", delta " << i;
			Assert::AreEqual(delta.Get(i), nablaB.Get(i), msg.str().c_str());
			for (uint32_t j = 0; j < 4; ++j)
			{
			  double expected = delta.Get(i) * previousLayerActivations.Get(j);
			  std::wostringstream msg;
			  msg << "Mismatch in mask " << mi << ", delta " << i << ", previous layer neuron " << j;
			  Assert::AreEqual(expected, nablaW.Get(i, j), 1e-5, msg.str().c_str());
			}
		  }
		  else
		  {
			std::wostringstream msg;
			msg << "Mismatch in mask " << mi << ", delta " << i;
			Assert::AreEqual(0.0, nablaB.Get(i), msg.str().c_str());
			for (uint32_t j = 0; j < 4; ++j)
			{
			  std::wostringstream msg;
			  msg << "Mismatch in mask " << mi << ", delta " << i << ", previous layer neuron " << j;
			  Assert::AreEqual(0.0, nablaW.Get(i, j), 1e-5, msg.str().c_str());
			}
		  }
		}
	  }
	}

	TEST_METHOD(FullyConnectedLayerUpdateWeightsAndBiases)
	{
	  auto weights = std::make_unique<Tensor>(std::initializer_list<double>{
		0.838504, 0.422149, 0.288635, 0.907155,
		  0.792704, 0.847105, 0.265283, 0.122859,
		  0.184963, 0.261111, 0.743236, 0.174590
	  }, 3, 4);
	  auto biases = std::make_unique<Tensor>(std::initializer_list<double>{ -0.1, 0.5, 0.0 });
	  Tensor nablaW(std::initializer_list<double>{
		-1.578502, 0.450965, -1.125088, 0.670935,
		  0.874675, -0.421671, 1.490070, 1.256670,
		  0.756746, 0.066774, 0.707417, -1.611769
	  }, 3, 4);
	  Tensor nablaB(std::initializer_list<double>{ 0.34, -0.667, 0.0});
	  double scalar = 0.1;

	  double expectedWeights[] = {
		0.996354, 0.377053, 0.401144, 0.840062,
		0.705237, 0.889272, 0.116276, -0.002808,
		0.109288, 0.254434, 0.672494, 0.335767
	  };
	  double expectedBiases[] = { -0.134, 0.5667, 0.0 };

	  FullyConnectedLayer layer(std::move(weights), std::move(biases), nullptr);

	  layer.UpdateWeightsAndBiases(nablaW, nablaB, scalar);

	  for (uint32_t i = 0; i < 12; ++i)
	  {
		Assert::AreEqual(expectedWeights[i], layer.Weights().Get(i), 1e-5);
	  }
	  for (uint32_t i = 0; i < 3; ++i)
	  {
		Assert::AreEqual(expectedBiases[i], layer.Biases().Get(i), 1e-5);
	  }
	}
  };
}
