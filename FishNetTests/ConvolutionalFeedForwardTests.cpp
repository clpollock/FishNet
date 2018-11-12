#include "stdafx.h"
#include "CppUnitTest.h"
#include "ConvolutionalLayer.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace ConvolutionalFeedForwardTests
{
  TEST_CLASS(ConvolutionalFeedForwardTests)
  {
  public:
	TEST_METHOD(SingleChannelSingle3x3FilterStride1ConvolutionalLayerFeedForward)
	{
	  double weights[9] =
	  {
		-0.838504,  0.422149,  0.288635,
		 0.792704, -0.847105,  0.265283,
		-0.184963,  0.261111, -0.743236
	  };
	  double bias = 0.1;
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 9), 1, 1, 3, 3);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>{ 0.1 });

	  Tensor input(std::initializer_list<double>{
		0.00544, -0.83437, -0.87491, -0.51001, 0.64589,
		  0.15090, 0.03972, -0.98896, -0.46934, -0.10584,
		  -0.45019, -0.43087, 0.59682, -0.66041, 0.06119,
		  0.27070, 0.69443, 0.88276, 0.63734, 0.84105,
		  -0.39515, -0.42934, -0.38812, 0.88456, 0.30129
	  }, 1, 5, 5);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 5, 5, 1, 0, nullptr);
	  Tensor output(1, 3, 3);
	  layer.FeedForward(input, output, nullptr);

	  for (uint32_t inputRow = 0; inputRow < 3; ++inputRow)
	  {
		for (uint32_t inputCol = 0; inputCol < 3; ++inputCol)
		{
		  double expectedActivation = 0.1; // bias
		  const double* weight = weights;
		  for (uint32_t weightRow = 0; weightRow < 3; ++weightRow)
		  {
			for (uint32_t weightCol = 0; weightCol < 3; ++weightCol)
			{
			  expectedActivation += (input.Get(0, inputRow + weightRow, inputCol + weightCol) * *weight);
			  ++weight;
			}
		  }
		  double actual = output.Get(0, inputRow, inputCol);
		  std::wostringstream msg;
		  msg << "Mismatch at row " << inputRow << ", column " << inputCol;
		  Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
		}
	  }
	}

	TEST_METHOD(SingleChannelSingle3x3FilterStride1ZeroPaddingConvolutionalLayerFeedForward)
	{
	  double weights[9] =
	  {
		-0.838504,  0.422149,  0.288635,
		 0.792704, -0.847105,  0.265283,
		-0.184963,  0.261111, -0.743236
	  };
	  double bias = 0.1;
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 9), 1, 1, 3, 3);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>{ 0.1 });

	  Tensor input(std::initializer_list<double>{
		0.00544, -0.83437, -0.87491, -0.51001, 0.64589,
		  0.15090, 0.03972, -0.98896, -0.46934, -0.10584,
		  -0.45019, -0.43087, 0.59682, -0.66041, 0.06119,
		  0.27070, 0.69443, 0.88276, 0.63734, 0.84105,
		  -0.39515, -0.42934, -0.38812, 0.88456, 0.30129
	  }, 1, 5, 5);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 5, 5, 1, 1, nullptr);
	  Tensor output(1, 5, 5);
	  layer.FeedForward(input, output, nullptr);
	  // Simple brute-force code. It's only a test so efficiency is not important.
	  for (int32_t inputRow = -1; inputRow < 4; ++inputRow)
	  {
		for (int32_t inputCol = -1; inputCol < 4; ++inputCol)
		{
		  double expectedActivation = 0.1; // bias
		  const double* weight = weights;
		  for (uint32_t weightRow = 0; weightRow < 3; ++weightRow)
		  {
			int32_t row = inputRow + weightRow;
			for (uint32_t weightCol = 0; weightCol < 3; ++weightCol)
			{
			  int32_t col = inputCol + weightCol;
			  if (row >= 0 && col >= 0 && row < 5 && col < 5)
				expectedActivation += (input.Get(0, row, col) * *weight);
			  ++weight;
			}
		  }
		  int32_t outputRow = inputRow + 1;
		  int32_t outputCol = inputCol + 1;
		  double actual = output.Get(0, outputRow, outputCol);
		  std::wostringstream msg;
		  msg << "Mismatch at row " << outputRow << ", column " << outputCol;
		  Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
		}
	  }
	}

	TEST_METHOD(SingleChannelSingle3x3FilterStride2ConvolutionalLayerFeedForward)
	{
	  double weights[9] =
	  {
		-0.838504,  0.422149,  0.288635,
		 0.792704, -0.847105,  0.265283,
		-0.184963,  0.261111, -0.743236
	  };
	  double bias = 0.1;
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 9), 1, 1, 3, 3);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>{ 0.1 });

	  Tensor input(std::initializer_list<double>{
		0.00544, -0.83437, -0.87491, -0.51001, 0.64589,
		  0.15090, 0.03972, -0.98896, -0.46934, -0.10584,
		  -0.45019, -0.43087, 0.59682, -0.66041, 0.06119,
		  0.27070, 0.69443, 0.88276, 0.63734, 0.84105,
		  -0.39515, -0.42934, -0.38812, 0.88456, 0.30129
	  }, 1, 5, 5);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 5, 5, 2, 0, nullptr);
	  Tensor output(1, 2, 2);
	  layer.FeedForward(input, output, nullptr);

	  uint32_t outputRow = 0;
	  for (uint32_t inputRow = 0; inputRow < 3; inputRow += 2)
	  {
		uint32_t outputCol = 0;
		for (uint32_t inputCol = 0; inputCol < 3; inputCol += 2)
		{
		  double expectedActivation = 0.1; // bias
		  const double* weight = weights;
		  for (uint32_t weightRow = 0; weightRow < 3; ++weightRow)
		  {
			for (uint32_t weightCol = 0; weightCol < 3; ++weightCol)
			{
			  expectedActivation += (input.Get(0, inputRow + weightRow, inputCol + weightCol) * *weight);
			  ++weight;
			}
		  }
		  double actual = output.Get(0, outputRow, outputCol);
		  std::wostringstream msg;
		  msg << "Mismatch at row " << outputRow << ", column " << outputCol;
		  Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
		  ++outputCol;
		}
		++outputRow;
	  }
	}

	TEST_METHOD(SingleChannelSingle3x3FilterStride2ZeroPaddingConvolutionalLayerFeedForward)
	{
	  double weights[9] =
	  {
		-0.838504,  0.422149,  0.288635,
		 0.792704, -0.847105,  0.265283,
		-0.184963,  0.261111, -0.743236
	  };
	  double bias = 0.1;
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 9), 1, 1, 3, 3);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>{ 0.1 });

	  Tensor input(std::initializer_list<double>{
		0.00544, -0.83437, -0.87491, -0.51001, 0.64589,
		  0.15090, 0.03972, -0.98896, -0.46934, -0.10584,
		  -0.45019, -0.43087, 0.59682, -0.66041, 0.06119,
		  0.27070, 0.69443, 0.88276, 0.63734, 0.84105,
		  -0.39515, -0.42934, -0.38812, 0.88456, 0.30129
	  }, 1, 5, 5);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 5, 5, 2, 2, nullptr);
	  Tensor output(1, 4, 4);
	  layer.FeedForward(input, output, nullptr);

	  uint32_t outputRow = 0;
	  for (int32_t inputRow = -2; inputRow < 5; inputRow += 2)
	  {
		uint32_t outputCol = 0;
		for (int32_t inputCol = -2; inputCol < 5; inputCol += 2)
		{
		  double expectedActivation = 0.1; // bias
		  const double* weight = weights;
		  for (int32_t weightRow = 0; weightRow < 3; ++weightRow)
		  {
			int32_t row = inputRow + weightRow;
			for (int32_t weightCol = 0; weightCol < 3; ++weightCol)
			{
			  int32_t col = inputCol + weightCol;
			  if (row >= 0 && col >=0 && row < 5 && col < 5)
				expectedActivation += (input.Get(0, row, col) * *weight);
			  ++weight;
			}
		  }
		  double actual = output.Get(0, outputRow, outputCol);
		  std::wostringstream msg;
		  msg << "Mismatch at row " << outputRow << ", column " << outputCol;
		  Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
		  ++outputCol;
		}
		++outputRow;
	  }
	}

	TEST_METHOD(SingleChannelTriple3x3FilterStride1ConvolutionalLayerFeedForward)
	{
	  double weights[27] =
	  {
		-0.94990, -0.73447, 0.41696,
		-0.70474, -0.11257, -0.53554,
		-0.39981, -0.12259, 0.26865,
		-0.72916, 0.40586, 0.07479,
		-0.43341, -0.97093, 0.49045,
		-0.24274, -0.30412, 0.01997,
		0.34119, 0.02124, 0.43289,
		-0.98409, 0.66278, -0.52050,
		-0.22532, 0.76755, -0.18896
	  };
	  double biases[3] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 27), 3, 1, 3, 3);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));

	  Tensor input(std::initializer_list<double>{
		0.00544, -0.83437, -0.87491, -0.51001, 0.64589,
		  0.15090, 0.03972, -0.98896, -0.46934, -0.10584,
		  -0.45019, -0.43087, 0.59682, -0.66041, 0.06119,
		  0.27070, 0.69443, 0.88276, 0.63734, 0.84105,
		  -0.39515, -0.42934, -0.38812, 0.88456, 0.30129
	  }, 1, 5, 5);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 5, 5, 1, 0, nullptr);
	  Tensor output(3, 3, 3);
	  layer.FeedForward(input, output, nullptr);

	  for (uint32_t filter = 0; filter < 3; ++filter)
	  {
		for (uint32_t inputRow = 0; inputRow < 3; ++inputRow)
		{
		  for (uint32_t inputCol = 0; inputCol < 3; ++inputCol)
		  {
			double expectedActivation = biases[filter];
			const double* weight = weights + (filter * 9);
			for (uint32_t weightRow = 0; weightRow < 3; ++weightRow)
			{
			  for (uint32_t weightCol = 0; weightCol < 3; ++weightCol)
			  {
				expectedActivation += (input.Get(0, inputRow + weightRow, inputCol + weightCol) * *weight);
				++weight;
			  }
			}
			double actual = output.Get(filter, inputRow, inputCol);
			std::wostringstream msg;
			msg << "Mismatch at filter " << filter << ", row " << inputRow << ", column " << inputCol;
			Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
		  }
		}
	  }
	}

	TEST_METHOD(SingleChannelTriple3x3FilterStride1ZeroPaddingConvolutionalLayerFeedForward)
	{
	  double weights[27] =
	  {
		-0.94990, -0.73447, 0.41696,
		-0.70474, -0.11257, -0.53554,
		-0.39981, -0.12259, 0.26865,
		-0.72916, 0.40586, 0.07479,
		-0.43341, -0.97093, 0.49045,
		-0.24274, -0.30412, 0.01997,
		0.34119, 0.02124, 0.43289,
		-0.98409, 0.66278, -0.52050,
		-0.22532, 0.76755, -0.18896
	  };
	  double biases[3] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 27), 3, 1, 3, 3);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));

	  Tensor input(std::initializer_list<double>{
		0.00544, -0.83437, -0.87491, -0.51001, 0.64589,
		  0.15090, 0.03972, -0.98896, -0.46934, -0.10584,
		  -0.45019, -0.43087, 0.59682, -0.66041, 0.06119,
		  0.27070, 0.69443, 0.88276, 0.63734, 0.84105,
		  -0.39515, -0.42934, -0.38812, 0.88456, 0.30129
	  }, 1, 5, 5);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 5, 5, 1, 1, nullptr);
	  Tensor output(3, 5, 5);
	  layer.FeedForward(input, output, nullptr);

	  for (uint32_t filter = 0; filter < 3; ++filter)
	  {
		for (uint32_t inputRow = -1; inputRow < 4; ++inputRow)
		{
		  for (uint32_t inputCol = -1; inputCol < 4; ++inputCol)
		  {
			double expectedActivation = biases[filter];
			const double* weight = weights + (filter * 9);
			for (uint32_t weightRow = 0; weightRow < 3; ++weightRow)
			{
			  int32_t row = inputRow + weightRow;
			  for (uint32_t weightCol = 0; weightCol < 3; ++weightCol)
			  {
				int32_t col = inputCol + weightCol;
				if (row >= 0 && col >=0 && row < 5 && col < 5)
				  expectedActivation += (input.Get(0, row, col) * *weight);
				++weight;
			  }
			}

			int32_t outputRow = inputRow + 1;
			int32_t outputCol = inputCol + 1;
			double actual = output.Get(filter, outputRow, outputCol);
			std::wostringstream msg;
			msg << "Mismatch at filter " << filter << ", row " << outputRow << ", column " << outputCol;
			Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
		  }
		}
	  }
	}

	TEST_METHOD(SingleChannelTriple3x3FilterStride2ConvolutionalLayerFeedForward)
	{
	  double weights[27] =
	  {
		-0.94990, -0.73447, 0.41696,
		-0.70474, -0.11257, -0.53554,
		-0.39981, -0.12259, 0.26865,
		-0.72916, 0.40586, 0.07479,
		-0.43341, -0.97093, 0.49045,
		-0.24274, -0.30412, 0.01997,
		0.34119, 0.02124, 0.43289,
		-0.98409, 0.66278, -0.52050,
		-0.22532, 0.76755, -0.18896
	  };
	  double biases[3] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 27), 3, 1, 3, 3);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));

	  Tensor input(std::initializer_list<double>{
		0.00544, -0.83437, -0.87491, -0.51001, 0.64589,
		0.15090, 0.03972, -0.98896, -0.46934, -0.10584,
		-0.45019, -0.43087, 0.59682, -0.66041, 0.06119,
		0.27070, 0.69443, 0.88276, 0.63734, 0.84105,
		-0.39515, -0.42934, -0.38812, 0.88456, 0.30129
	  }, 1, 5, 5);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 5, 5, 2, 0, nullptr);
	  Tensor output(3, 2, 2);
	  layer.FeedForward(input, output, nullptr);

	  for (uint32_t filter = 0; filter < 3; ++filter)
	  {
		for (uint32_t inputRow = 0; inputRow <= 2; inputRow += 2)
		{
		  for (uint32_t inputCol = 0; inputCol <= 2; ++inputCol += 2)
		  {
			double expectedActivation = biases[filter];
			const double* weight = weights + (filter * 9);
			for (uint32_t weightRow = 0; weightRow < 3; ++weightRow)
			{
			  for (uint32_t weightCol = 0; weightCol < 3; ++weightCol)
			  {
				expectedActivation += (input.Get(0, inputRow + weightRow, inputCol + weightCol) * *weight);
				++weight;
			  }
			}
			uint32_t outputRow = inputRow / 2;
			uint32_t outputCol = inputCol / 2;
			double actual = output.Get(filter, outputRow, outputCol);
			std::wostringstream msg;
			msg << "Mismatch at filter " << filter << ", row " << outputRow << ", column " << outputCol;
			Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
		  }
		}
	  }
	}

	TEST_METHOD(SingleChannelTriple3x3FilterStride2ZeroPaddingConvolutionalLayerFeedForward)
	{
	  double weights[27] =
	  {
		-0.94990, -0.73447, 0.41696,
		-0.70474, -0.11257, -0.53554,
		-0.39981, -0.12259, 0.26865,
		-0.72916, 0.40586, 0.07479,
		-0.43341, -0.97093, 0.49045,
		-0.24274, -0.30412, 0.01997,
		0.34119, 0.02124, 0.43289,
		-0.98409, 0.66278, -0.52050,
		-0.22532, 0.76755, -0.18896
	  };
	  double biases[3] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 27), 3, 1, 3, 3);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));

	  Tensor input(std::initializer_list<double>{
		0.00544, -0.83437, -0.87491, -0.51001, 0.64589,
		0.15090, 0.03972, -0.98896, -0.46934, -0.10584,
		-0.45019, -0.43087, 0.59682, -0.66041, 0.06119,
		0.27070, 0.69443, 0.88276, 0.63734, 0.84105,
		-0.39515, -0.42934, -0.38812, 0.88456, 0.30129
	  }, 1, 5, 5);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 5, 5, 2, 2, nullptr);
	  Tensor output(3, 4, 4);
	  layer.FeedForward(input, output, nullptr);

	  for (uint32_t filter = 0; filter < 3; ++filter)
	  {
		int32_t outputRow = 0;
		for (int32_t inputRow = -2; inputRow < 5; inputRow += 2)
		{
		  int32_t outputCol = 0;
		  for (int32_t inputCol = -2; inputCol < 5; inputCol += 2)
		  {
			double expectedActivation = biases[filter];
			const double* weight = weights + (filter * 9);
			for (int32_t weightRow = 0; weightRow < 3; ++weightRow)
			{
			  int32_t row = inputRow + weightRow;
			  for (int32_t weightCol = 0; weightCol < 3; ++weightCol)
			  {
				int32_t col = inputCol + weightCol;
				if (row >= 0 && col >=0 && row < 5 && col < 5)
				  expectedActivation += (input.Get(0, row, col) * *weight);
				++weight;
			  }
			}
			double actual = output.Get(filter, outputRow, outputCol);
			std::wostringstream msg;
			msg << "Mismatch at filter " << filter << ", row " << outputRow << ", column " << outputCol;
			Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
			outputCol += 1;
		  }
		  outputRow += 1;
		}
	  }
	}

	TEST_METHOD(SingleChannelTriple4x4FilterStride1ConvolutionalLayerFeedForward)
	{
	  double weights[48] =
	  {
		0.32238, 0.98944, 0.63056, 0.31580,
		-0.38978, -0.67950, 0.06071, -0.71658,
		-0.77753, 0.06306, 0.10508, 0.95284,
		0.46683, -0.27453, -0.60782, -0.57690,
		-0.99589, 0.52354, 0.47561, 0.40587,
		-0.35232, 0.10972, -0.33507, -0.54053,
		0.78707, -0.20374, -0.78467, -0.91656,
		0.44568, 0.18876, -0.85019, 0.49560,
		0.18865, -0.08489, 0.24662, -0.43657,
		-0.34598, -0.04193, 0.87457, 0.63628,
		-0.68881, 0.03971, 0.29993, -0.11905,
		-0.69384, -0.33466, -0.73462, 0.87781
	  };
	  double biases[] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 48), 3, 1, 4, 4);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));

	  Tensor input(std::initializer_list<double>{
		0.00544, -0.83437, -0.87491, -0.51001, 0.64589,
		  0.15090, 0.03972, -0.98896, -0.46934, -0.10584,
		  -0.45019, -0.43087, 0.59682, -0.66041, 0.06119,
		  0.27070, 0.69443, 0.88276, 0.63734, 0.84105,
		  -0.39515, -0.42934, -0.38812, 0.88456, 0.30129
	  }, 1, 5, 5);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 5, 5, 1, 0, nullptr);
	  Tensor output(3, 2, 2);
	  layer.FeedForward(input, output, nullptr);

	  for (uint32_t filter = 0; filter < 3; ++filter)
	  {
		for (uint32_t inputRow = 0; inputRow < 2; ++inputRow)
		{
		  for (uint32_t inputCol = 0; inputCol < 2; ++inputCol)
		  {
			double expectedActivation = biases[filter];
			const double* weight = weights + (filter * 16);
			for (uint32_t weightRow = 0; weightRow < 4; ++weightRow)
			{
			  for (uint32_t weightCol = 0; weightCol < 4; ++weightCol)
			  {
				expectedActivation += (input.Get(0, inputRow + weightRow, inputCol + weightCol) * *weight);
				++weight;
			  }
			}
			double actual = output.Get(filter, inputRow, inputCol);
			std::wostringstream msg;
			msg << "Mismatch at filter " << filter << ", row " << inputRow << ", column " << inputCol;
			Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
		  }
		}
	  }
	}

	TEST_METHOD(SingleChannelTriple4x4FilterStride1ZeroPaddingConvolutionalLayerFeedForward)
	{
	  double weights[48] =
	  {
		0.32238, 0.98944, 0.63056, 0.31580,
		-0.38978, -0.67950, 0.06071, -0.71658,
		-0.77753, 0.06306, 0.10508, 0.95284,
		0.46683, -0.27453, -0.60782, -0.57690,
		-0.99589, 0.52354, 0.47561, 0.40587,
		-0.35232, 0.10972, -0.33507, -0.54053,
		0.78707, -0.20374, -0.78467, -0.91656,
		0.44568, 0.18876, -0.85019, 0.49560,
		0.18865, -0.08489, 0.24662, -0.43657,
		-0.34598, -0.04193, 0.87457, 0.63628,
		-0.68881, 0.03971, 0.29993, -0.11905,
		-0.69384, -0.33466, -0.73462, 0.87781
	  };
	  double biases[] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 48), 3, 1, 4, 4);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));

	  Tensor input(std::initializer_list<double>{
		0.00544, -0.83437, -0.87491, -0.51001, 0.64589,
		  0.15090, 0.03972, -0.98896, -0.46934, -0.10584,
		  -0.45019, -0.43087, 0.59682, -0.66041, 0.06119,
		  0.27070, 0.69443, 0.88276, 0.63734, 0.84105,
		  -0.39515, -0.42934, -0.38812, 0.88456, 0.30129
	  }, 1, 5, 5);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 5, 5, 1, 1, nullptr);
	  Tensor output(3, 4, 4);
	  layer.FeedForward(input, output, nullptr);

	  for (uint32_t filter = 0; filter < 3; ++filter)
	  {
		int32_t outputRow = 0;
		for (int32_t inputRow = -1; inputRow < 3; ++inputRow)
		{
		  int32_t outputCol = 0;
		  for (int32_t inputCol = -1; inputCol < 3; ++inputCol)
		  {
			double expectedActivation = biases[filter];
			const double* weight = weights + (filter * 16);
			for (int32_t weightRow = 0; weightRow < 4; ++weightRow)
			{
			  int32_t row = inputRow + weightRow;
			  for (int32_t weightCol = 0; weightCol < 4; ++weightCol)
			  {
				int32_t col = inputCol + weightCol;
				if (row >= 0 && col >= 0 && row < 5 && col < 5)
				  expectedActivation += (input.Get(0, row, col) * *weight);
				++weight;
			  }
			}
			double actual = output.Get(filter, outputRow, outputCol);
			std::wostringstream msg;
			msg << "Mismatch at filter " << filter << ", row " << outputRow << ", column " << outputCol;
			Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
			++outputCol;
		  }
		  ++outputRow;
		}
	  }
	}

	TEST_METHOD(FourChannelTriple4x4FilterStride1ConvolutionalLayerFeedForward)
	{
	  double weights[3 * 4 * 4 * 4] =
	  {
		-0.42190, -0.07073, -0.37965, -0.50923,
		0.82046, -0.40299, 0.43058, 0.26103,
		0.48791, 0.75313, -0.38173, 0.77909,
		-0.28330, 0.59229, 0.33269, -0.04618,
		-0.78984, -0.75417, -0.95192, -0.55587,
		-0.10078, 0.42707, -0.57210, -0.70336,
		-0.35389, 0.72271, 0.98835, 0.26764,
		0.32582, -0.26176, -0.55671, -0.42471,
		-0.30906, 0.05194, -0.84092, 0.79222,
		0.27393, -0.63657, -0.64110, 0.34900,
		0.96608, 0.80632, 0.29210, 0.79523,
		0.59684, -0.52751, -0.13679, -0.27715,
		-0.65315, -0.03183, 0.98749, 0.62926,
		0.78129, 0.73793, 0.97178, 0.52113,
		0.38966, -0.47312, -0.75155, 0.13760,
		-0.30766, 0.79871, 0.37302, 0.13712,
		-0.98138, -0.32996, -0.47085, 0.32317,
		-0.46456, -0.60033, 0.53550, -0.45387,
		-0.56226, -0.16800, -0.14419, -0.42532,
		0.14565, 0.04163, 0.14524, 0.53292,
		-0.98047, -0.96334, -0.65149, 0.47023,
		-0.51596, 0.20417, -0.69487, 0.97148,
		-0.47098, 0.53511, -0.02126, -0.44055,
		-0.48167, 0.09429, -0.15374, 0.34258,
		-0.93103, 0.22103, -0.99384, 0.99203,
		-0.66201, 0.29653, 0.90861, -0.09646,
		0.47369, -0.14171, 0.53700, 0.05467,
		0.99783, 0.01265, 0.15270, -0.36204,
		-0.26809, 0.28557, 0.71028, -0.98100,
		0.78682, -0.25404, -0.57801, -0.35418,
		-0.69634, -0.52972, 0.00478, -0.66064,
		0.03603, 0.03421, -0.44179, -0.92546,
		-0.32836, 0.26241, 0.88489, -0.06405,
		0.11215, -0.83351, 0.25491, 0.21188,
		0.31244, -0.65995, 0.64232, -0.10133,
		0.93793, 0.74544, -0.28907, -0.89644,
		0.25116, -0.67054, 0.02529, 0.39685,
		-0.49114, 0.16491, -0.13570, -0.65323,
		-0.51811, 0.15955, -0.58550, 0.96218,
		-0.85121, -0.93031, -0.75497, -0.72648,
		0.42119, -0.82081, -0.94288, -0.58808,
		-0.85918, -0.27473, 0.03149, -0.79443,
		-0.93598, -0.00243, 0.02433, -0.65511,
		0.64065, -0.11250, 0.82651, 0.41035,
		0.60832, 0.73556, 0.39781, -0.00366,
		-0.00015, 0.62124, 0.13339, -0.02858,
		0.37242, -0.25726, 0.42854, 0.88884,
		0.20685, 0.83611, -0.59334, -0.28102
	  };
	  double biases[] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 3 * 4 * 4 * 4), 3, 4, 4, 4);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));

	  Tensor input(std::initializer_list<double>{
		0.93257, -0.69348, 0.88512, 0.15497, -0.85682, 0.95709,
		-0.76236, -0.87117, -0.69342, 0.06064, 0.13694, -0.11313,
		-0.41186, -0.47179, -0.95423, -0.43988, -0.63668, 0.85254,
		-0.43114, 0.34758, -0.05468, 0.62974, -0.78308, 0.64675,
		-0.33506, 0.90725, -0.71736, -0.53904, -0.81525, -0.30506,
		0.36369, -0.66749, 0.71084, -0.78597, 0.71073, 0.38355,
		-0.48088, 0.85047, 0.45972, -0.09723, -0.00188, 0.45151,
		-0.86415, 0.38864, 0.42819, -0.23813, -0.78795, 0.82886,
		-0.54423, 0.51731, 0.47529, 0.95405, -0.89665, -0.35599,
		0.28404, 0.09745, -0.09898, 0.67697, 0.06644, -0.42143,
		-0.17749, 0.16599, 0.69045, 0.85710, 0.66507, 0.65792,
		-0.68536, 0.88013, 0.54331, -0.45803, -0.93451, -0.46768,
		-0.80612, 0.28542, 0.93249, -0.03669, -0.59803, 0.24587,
		0.99741, -0.83093, -0.61395, -0.36269, 0.76147, 0.91036,
		0.93110, 0.94640, 0.56900, 0.25906, -0.48510, 0.84585,
		-0.02996, 0.05263, 0.13543, 0.68872, 0.15700, 0.98693,
		0.62692, 0.48035, 0.62869, 0.08488, 0.63225, -0.41932,
		-0.85007, 0.91191, -0.73345, 0.11369, -0.26750, -0.73507,
		-0.79675, 0.39817, 0.08518, -0.89688, -0.46205, 0.50970,
		-0.98583, 0.55502, -0.95421, 0.85380, -0.82145, 0.66815,
		0.69849, 0.43104, 0.37487, 0.18473, 0.18205, -0.19230,
		0.70224, 0.85945, 0.60625, 0.01480, 0.82155, 0.68235,
		0.79858, -0.50783, 0.50093, -0.39160, -0.86973, 0.27414,
		-0.93758, 0.66361, 0.61068, 0.53770, -0.60279, 0.33966
	  }, 4, 6, 6);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 6, 6, 1, 0, nullptr);
	  Tensor output(3, 3, 3);
	  layer.FeedForward(input, output, nullptr);

	  for (uint32_t filter = 0; filter < 3; ++filter)
	  {
		for (uint32_t outputRow = 0; outputRow < 3; ++outputRow)
		{
		  for (uint32_t outputCol = 0; outputCol < 3; ++outputCol)
		  {
			double expectedActivation = biases[filter];
			const double* weight = weights + (filter * 16 * 4);
			for (uint32_t inputChannel = 0; inputChannel < 4; ++inputChannel)
			{
			  for (uint32_t weightRow = 0; weightRow < 4; ++weightRow)
			  {
				for (uint32_t weightCol = 0; weightCol < 4; ++weightCol)
				{
				  expectedActivation += (input.Get(inputChannel, outputRow + weightRow, outputCol + weightCol) * *weight);
				  ++weight;
				}
			  }
			}
			double actual = output.Get(filter, outputRow, outputCol);
			std::wostringstream msg;
			msg << "Mismatch at filter " << filter << ", row " << outputRow << ", column " << outputCol;
			Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
		  }
		}
	  }
	}

	TEST_METHOD(FourChannelTriple4x4FilterStride1ZeroPaddingConvolutionalLayerFeedForward)
	{
	  double weights[3 * 4 * 4 * 4] =
	  {
		-0.42190, -0.07073, -0.37965, -0.50923,
		0.82046, -0.40299, 0.43058, 0.26103,
		0.48791, 0.75313, -0.38173, 0.77909,
		-0.28330, 0.59229, 0.33269, -0.04618,
		-0.78984, -0.75417, -0.95192, -0.55587,
		-0.10078, 0.42707, -0.57210, -0.70336,
		-0.35389, 0.72271, 0.98835, 0.26764,
		0.32582, -0.26176, -0.55671, -0.42471,
		-0.30906, 0.05194, -0.84092, 0.79222,
		0.27393, -0.63657, -0.64110, 0.34900,
		0.96608, 0.80632, 0.29210, 0.79523,
		0.59684, -0.52751, -0.13679, -0.27715,
		-0.65315, -0.03183, 0.98749, 0.62926,
		0.78129, 0.73793, 0.97178, 0.52113,
		0.38966, -0.47312, -0.75155, 0.13760,
		-0.30766, 0.79871, 0.37302, 0.13712,
		-0.98138, -0.32996, -0.47085, 0.32317,
		-0.46456, -0.60033, 0.53550, -0.45387,
		-0.56226, -0.16800, -0.14419, -0.42532,
		0.14565, 0.04163, 0.14524, 0.53292,
		-0.98047, -0.96334, -0.65149, 0.47023,
		-0.51596, 0.20417, -0.69487, 0.97148,
		-0.47098, 0.53511, -0.02126, -0.44055,
		-0.48167, 0.09429, -0.15374, 0.34258,
		-0.93103, 0.22103, -0.99384, 0.99203,
		-0.66201, 0.29653, 0.90861, -0.09646,
		0.47369, -0.14171, 0.53700, 0.05467,
		0.99783, 0.01265, 0.15270, -0.36204,
		-0.26809, 0.28557, 0.71028, -0.98100,
		0.78682, -0.25404, -0.57801, -0.35418,
		-0.69634, -0.52972, 0.00478, -0.66064,
		0.03603, 0.03421, -0.44179, -0.92546,
		-0.32836, 0.26241, 0.88489, -0.06405,
		0.11215, -0.83351, 0.25491, 0.21188,
		0.31244, -0.65995, 0.64232, -0.10133,
		0.93793, 0.74544, -0.28907, -0.89644,
		0.25116, -0.67054, 0.02529, 0.39685,
		-0.49114, 0.16491, -0.13570, -0.65323,
		-0.51811, 0.15955, -0.58550, 0.96218,
		-0.85121, -0.93031, -0.75497, -0.72648,
		0.42119, -0.82081, -0.94288, -0.58808,
		-0.85918, -0.27473, 0.03149, -0.79443,
		-0.93598, -0.00243, 0.02433, -0.65511,
		0.64065, -0.11250, 0.82651, 0.41035,
		0.60832, 0.73556, 0.39781, -0.00366,
		-0.00015, 0.62124, 0.13339, -0.02858,
		0.37242, -0.25726, 0.42854, 0.88884,
		0.20685, 0.83611, -0.59334, -0.28102
	  };
	  double biases[] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 3 * 4 * 4 * 4), 3, 4, 4, 4);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));

	  Tensor input(std::initializer_list<double>{
		0.93257, -0.69348, 0.88512, 0.15497, -0.85682, 0.95709,
		-0.76236, -0.87117, -0.69342, 0.06064, 0.13694, -0.11313,
		-0.41186, -0.47179, -0.95423, -0.43988, -0.63668, 0.85254,
		-0.43114, 0.34758, -0.05468, 0.62974, -0.78308, 0.64675,
		-0.33506, 0.90725, -0.71736, -0.53904, -0.81525, -0.30506,
		0.36369, -0.66749, 0.71084, -0.78597, 0.71073, 0.38355,
		-0.48088, 0.85047, 0.45972, -0.09723, -0.00188, 0.45151,
		-0.86415, 0.38864, 0.42819, -0.23813, -0.78795, 0.82886,
		-0.54423, 0.51731, 0.47529, 0.95405, -0.89665, -0.35599,
		0.28404, 0.09745, -0.09898, 0.67697, 0.06644, -0.42143,
		-0.17749, 0.16599, 0.69045, 0.85710, 0.66507, 0.65792,
		-0.68536, 0.88013, 0.54331, -0.45803, -0.93451, -0.46768,
		-0.80612, 0.28542, 0.93249, -0.03669, -0.59803, 0.24587,
		0.99741, -0.83093, -0.61395, -0.36269, 0.76147, 0.91036,
		0.93110, 0.94640, 0.56900, 0.25906, -0.48510, 0.84585,
		-0.02996, 0.05263, 0.13543, 0.68872, 0.15700, 0.98693,
		0.62692, 0.48035, 0.62869, 0.08488, 0.63225, -0.41932,
		-0.85007, 0.91191, -0.73345, 0.11369, -0.26750, -0.73507,
		-0.79675, 0.39817, 0.08518, -0.89688, -0.46205, 0.50970,
		-0.98583, 0.55502, -0.95421, 0.85380, -0.82145, 0.66815,
		0.69849, 0.43104, 0.37487, 0.18473, 0.18205, -0.19230,
		0.70224, 0.85945, 0.60625, 0.01480, 0.82155, 0.68235,
		0.79858, -0.50783, 0.50093, -0.39160, -0.86973, 0.27414,
		-0.93758, 0.66361, 0.61068, 0.53770, -0.60279, 0.33966
	  }, 4, 6, 6);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 6, 6, 1, 1, nullptr);
	  Tensor output(3, 5, 5);
	  layer.FeedForward(input, output, nullptr);

	  for (uint32_t filter = 0; filter < 3; ++filter)
	  {
		for (int32_t inputRow = -1; inputRow < 3; ++inputRow)
		{
		  for (int32_t inputCol = -1; inputCol < 3; ++inputCol)
		  {
			double expectedActivation = biases[filter];
			const double* weight = weights + (filter * 16 * 4);
			for (uint32_t inputChannel = 0; inputChannel < 4; ++inputChannel)
			{
			  for (int32_t weightRow = 0; weightRow < 4; ++weightRow)
			  {
  				int32_t row = inputRow + weightRow;
				for (int32_t weightCol = 0; weightCol < 4; ++weightCol)
				{
				  int32_t col = inputCol + weightCol;
				  if (row >= 0 && col >= 0 && row < 6 && col < 6)
					expectedActivation += (input.Get(inputChannel, row, col) * *weight);
				  ++weight;
				}
			  }
			}
			int32_t outputRow = inputRow + 1;
			int32_t outputCol = inputCol + 1;
			double actual = output.Get(filter, outputRow, outputCol);
			std::wostringstream msg;
			msg << "Mismatch at filter " << filter << ", row " << outputRow << ", column " << outputCol;
			Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
		  }
		}
	  }
	}

	TEST_METHOD(FourChannelTriple4x4FilterStride2ConvolutionalLayerFeedForward)
	{
	  double weights[3 * 4 * 4 * 4] =
	  {
		-0.42190, -0.07073, -0.37965, -0.50923,
		0.82046, -0.40299, 0.43058, 0.26103,
		0.48791, 0.75313, -0.38173, 0.77909,
		-0.28330, 0.59229, 0.33269, -0.04618,
		-0.78984, -0.75417, -0.95192, -0.55587,
		-0.10078, 0.42707, -0.57210, -0.70336,
		-0.35389, 0.72271, 0.98835, 0.26764,
		0.32582, -0.26176, -0.55671, -0.42471,
		-0.30906, 0.05194, -0.84092, 0.79222,
		0.27393, -0.63657, -0.64110, 0.34900,
		0.96608, 0.80632, 0.29210, 0.79523,
		0.59684, -0.52751, -0.13679, -0.27715,
		-0.65315, -0.03183, 0.98749, 0.62926,
		0.78129, 0.73793, 0.97178, 0.52113,
		0.38966, -0.47312, -0.75155, 0.13760,
		-0.30766, 0.79871, 0.37302, 0.13712,
		-0.98138, -0.32996, -0.47085, 0.32317,
		-0.46456, -0.60033, 0.53550, -0.45387,
		-0.56226, -0.16800, -0.14419, -0.42532,
		0.14565, 0.04163, 0.14524, 0.53292,
		-0.98047, -0.96334, -0.65149, 0.47023,
		-0.51596, 0.20417, -0.69487, 0.97148,
		-0.47098, 0.53511, -0.02126, -0.44055,
		-0.48167, 0.09429, -0.15374, 0.34258,
		-0.93103, 0.22103, -0.99384, 0.99203,
		-0.66201, 0.29653, 0.90861, -0.09646,
		0.47369, -0.14171, 0.53700, 0.05467,
		0.99783, 0.01265, 0.15270, -0.36204,
		-0.26809, 0.28557, 0.71028, -0.98100,
		0.78682, -0.25404, -0.57801, -0.35418,
		-0.69634, -0.52972, 0.00478, -0.66064,
		0.03603, 0.03421, -0.44179, -0.92546,
		-0.32836, 0.26241, 0.88489, -0.06405,
		0.11215, -0.83351, 0.25491, 0.21188,
		0.31244, -0.65995, 0.64232, -0.10133,
		0.93793, 0.74544, -0.28907, -0.89644,
		0.25116, -0.67054, 0.02529, 0.39685,
		-0.49114, 0.16491, -0.13570, -0.65323,
		-0.51811, 0.15955, -0.58550, 0.96218,
		-0.85121, -0.93031, -0.75497, -0.72648,
		0.42119, -0.82081, -0.94288, -0.58808,
		-0.85918, -0.27473, 0.03149, -0.79443,
		-0.93598, -0.00243, 0.02433, -0.65511,
		0.64065, -0.11250, 0.82651, 0.41035,
		0.60832, 0.73556, 0.39781, -0.00366,
		-0.00015, 0.62124, 0.13339, -0.02858,
		0.37242, -0.25726, 0.42854, 0.88884,
		0.20685, 0.83611, -0.59334, -0.28102
	  };
	  double biases[] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 3 * 4 * 4 * 4), 3, 4, 4, 4);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));

	  Tensor input(std::initializer_list<double>{
		0.93257, -0.69348, 0.88512, 0.15497, -0.85682, 0.95709,
		-0.76236, -0.87117, -0.69342, 0.06064, 0.13694, -0.11313,
		-0.41186, -0.47179, -0.95423, -0.43988, -0.63668, 0.85254,
		-0.43114, 0.34758, -0.05468, 0.62974, -0.78308, 0.64675,
		-0.33506, 0.90725, -0.71736, -0.53904, -0.81525, -0.30506,
		0.36369, -0.66749, 0.71084, -0.78597, 0.71073, 0.38355,
		-0.48088, 0.85047, 0.45972, -0.09723, -0.00188, 0.45151,
		-0.86415, 0.38864, 0.42819, -0.23813, -0.78795, 0.82886,
		-0.54423, 0.51731, 0.47529, 0.95405, -0.89665, -0.35599,
		0.28404, 0.09745, -0.09898, 0.67697, 0.06644, -0.42143,
		-0.17749, 0.16599, 0.69045, 0.85710, 0.66507, 0.65792,
		-0.68536, 0.88013, 0.54331, -0.45803, -0.93451, -0.46768,
		-0.80612, 0.28542, 0.93249, -0.03669, -0.59803, 0.24587,
		0.99741, -0.83093, -0.61395, -0.36269, 0.76147, 0.91036,
		0.93110, 0.94640, 0.56900, 0.25906, -0.48510, 0.84585,
		-0.02996, 0.05263, 0.13543, 0.68872, 0.15700, 0.98693,
		0.62692, 0.48035, 0.62869, 0.08488, 0.63225, -0.41932,
		-0.85007, 0.91191, -0.73345, 0.11369, -0.26750, -0.73507,
		-0.79675, 0.39817, 0.08518, -0.89688, -0.46205, 0.50970,
		-0.98583, 0.55502, -0.95421, 0.85380, -0.82145, 0.66815,
		0.69849, 0.43104, 0.37487, 0.18473, 0.18205, -0.19230,
		0.70224, 0.85945, 0.60625, 0.01480, 0.82155, 0.68235,
		0.79858, -0.50783, 0.50093, -0.39160, -0.86973, 0.27414,
		-0.93758, 0.66361, 0.61068, 0.53770, -0.60279, 0.33966
	  }, 4, 6, 6);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 6, 6, 2, 0, nullptr);
	  Tensor output(3, 2, 2);
	  layer.FeedForward(input, output, nullptr);

	  for (uint32_t filter = 0; filter < 3; ++filter)
	  {
		for (uint32_t outputRow = 0; outputRow < 2; ++outputRow)
		{
		  for (uint32_t outputCol = 0; outputCol < 2; ++outputCol)
		  {
			double expectedActivation = biases[filter];
			const double* weight = weights + (filter * 16 * 4);
			for (uint32_t inputChannel = 0; inputChannel < 4; ++inputChannel)
			{
			  for (uint32_t weightRow = 0; weightRow < 4; ++weightRow)
			  {
				for (uint32_t weightCol = 0; weightCol < 4; ++weightCol)
				{
				  uint32_t inputRow = outputRow * 2 + weightRow;
				  uint32_t inputCol = outputCol * 2 + weightCol;
				  expectedActivation += (input.Get(inputChannel, inputRow, inputCol) * *weight);
				  ++weight;
				}
			  }
			}
			double actual = output.Get(filter, outputRow, outputCol);
			std::wostringstream msg;
			msg << "Mismatch at filter " << filter << ", row " << outputRow << ", column " << outputCol;
			Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
		  }
		}
	  }
	}

	TEST_METHOD(FourChannelTriple4x4FilterStride2ZeroPaddingConvolutionalLayerFeedForward)
	{
	  double weights[3 * 4 * 4 * 4] =
	  {
		-0.42190, -0.07073, -0.37965, -0.50923,
		0.82046, -0.40299, 0.43058, 0.26103,
		0.48791, 0.75313, -0.38173, 0.77909,
		-0.28330, 0.59229, 0.33269, -0.04618,
		-0.78984, -0.75417, -0.95192, -0.55587,
		-0.10078, 0.42707, -0.57210, -0.70336,
		-0.35389, 0.72271, 0.98835, 0.26764,
		0.32582, -0.26176, -0.55671, -0.42471,
		-0.30906, 0.05194, -0.84092, 0.79222,
		0.27393, -0.63657, -0.64110, 0.34900,
		0.96608, 0.80632, 0.29210, 0.79523,
		0.59684, -0.52751, -0.13679, -0.27715,
		-0.65315, -0.03183, 0.98749, 0.62926,
		0.78129, 0.73793, 0.97178, 0.52113,
		0.38966, -0.47312, -0.75155, 0.13760,
		-0.30766, 0.79871, 0.37302, 0.13712,
		-0.98138, -0.32996, -0.47085, 0.32317,
		-0.46456, -0.60033, 0.53550, -0.45387,
		-0.56226, -0.16800, -0.14419, -0.42532,
		0.14565, 0.04163, 0.14524, 0.53292,
		-0.98047, -0.96334, -0.65149, 0.47023,
		-0.51596, 0.20417, -0.69487, 0.97148,
		-0.47098, 0.53511, -0.02126, -0.44055,
		-0.48167, 0.09429, -0.15374, 0.34258,
		-0.93103, 0.22103, -0.99384, 0.99203,
		-0.66201, 0.29653, 0.90861, -0.09646,
		0.47369, -0.14171, 0.53700, 0.05467,
		0.99783, 0.01265, 0.15270, -0.36204,
		-0.26809, 0.28557, 0.71028, -0.98100,
		0.78682, -0.25404, -0.57801, -0.35418,
		-0.69634, -0.52972, 0.00478, -0.66064,
		0.03603, 0.03421, -0.44179, -0.92546,
		-0.32836, 0.26241, 0.88489, -0.06405,
		0.11215, -0.83351, 0.25491, 0.21188,
		0.31244, -0.65995, 0.64232, -0.10133,
		0.93793, 0.74544, -0.28907, -0.89644,
		0.25116, -0.67054, 0.02529, 0.39685,
		-0.49114, 0.16491, -0.13570, -0.65323,
		-0.51811, 0.15955, -0.58550, 0.96218,
		-0.85121, -0.93031, -0.75497, -0.72648,
		0.42119, -0.82081, -0.94288, -0.58808,
		-0.85918, -0.27473, 0.03149, -0.79443,
		-0.93598, -0.00243, 0.02433, -0.65511,
		0.64065, -0.11250, 0.82651, 0.41035,
		0.60832, 0.73556, 0.39781, -0.00366,
		-0.00015, 0.62124, 0.13339, -0.02858,
		0.37242, -0.25726, 0.42854, 0.88884,
		0.20685, 0.83611, -0.59334, -0.28102
	  };
	  double biases[] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 3 * 4 * 4 * 4), 3, 4, 4, 4);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));

	  Tensor input(std::initializer_list<double>{
		0.93257, -0.69348, 0.88512, 0.15497, -0.85682, 0.95709,
		-0.76236, -0.87117, -0.69342, 0.06064, 0.13694, -0.11313,
		-0.41186, -0.47179, -0.95423, -0.43988, -0.63668, 0.85254,
		-0.43114, 0.34758, -0.05468, 0.62974, -0.78308, 0.64675,
		-0.33506, 0.90725, -0.71736, -0.53904, -0.81525, -0.30506,
		0.36369, -0.66749, 0.71084, -0.78597, 0.71073, 0.38355,
		-0.48088, 0.85047, 0.45972, -0.09723, -0.00188, 0.45151,
		-0.86415, 0.38864, 0.42819, -0.23813, -0.78795, 0.82886,
		-0.54423, 0.51731, 0.47529, 0.95405, -0.89665, -0.35599,
		0.28404, 0.09745, -0.09898, 0.67697, 0.06644, -0.42143,
		-0.17749, 0.16599, 0.69045, 0.85710, 0.66507, 0.65792,
		-0.68536, 0.88013, 0.54331, -0.45803, -0.93451, -0.46768,
		-0.80612, 0.28542, 0.93249, -0.03669, -0.59803, 0.24587,
		0.99741, -0.83093, -0.61395, -0.36269, 0.76147, 0.91036,
		0.93110, 0.94640, 0.56900, 0.25906, -0.48510, 0.84585,
		-0.02996, 0.05263, 0.13543, 0.68872, 0.15700, 0.98693,
		0.62692, 0.48035, 0.62869, 0.08488, 0.63225, -0.41932,
		-0.85007, 0.91191, -0.73345, 0.11369, -0.26750, -0.73507,
		-0.79675, 0.39817, 0.08518, -0.89688, -0.46205, 0.50970,
		-0.98583, 0.55502, -0.95421, 0.85380, -0.82145, 0.66815,
		0.69849, 0.43104, 0.37487, 0.18473, 0.18205, -0.19230,
		0.70224, 0.85945, 0.60625, 0.01480, 0.82155, 0.68235,
		0.79858, -0.50783, 0.50093, -0.39160, -0.86973, 0.27414,
		-0.93758, 0.66361, 0.61068, 0.53770, -0.60279, 0.33966
	  }, 4, 6, 6);

	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 6, 6, 2, 1, nullptr);
	  Tensor output(3, 3, 3);
	  layer.FeedForward(input, output, nullptr);

	  for (uint32_t filter = 0; filter < 3; ++filter)
	  {
		for (uint32_t outputRow = 0; outputRow < 3; ++outputRow)
		{
		  for (uint32_t outputCol = 0; outputCol < 3; ++outputCol)
		  {
			double expectedActivation = biases[filter];
			const double* weight = weights + (filter * 16 * 4);
			for (uint32_t inputChannel = 0; inputChannel < 4; ++inputChannel)
			{
			  for (uint32_t weightRow = 0; weightRow < 4; ++weightRow)
			  {
				for (uint32_t weightCol = 0; weightCol < 4; ++weightCol)
				{
				  uint32_t inputRow = outputRow * 2 + weightRow - 1;
				  uint32_t inputCol = outputCol * 2 + weightCol - 1;
				  if (inputRow >= 0 && inputCol >= 0 && inputRow < 6 && inputCol < 6)
					expectedActivation += (input.Get(inputChannel, inputRow, inputCol) * *weight);
				  ++weight;
				}
			  }
			}
			double actual = output.Get(filter, outputRow, outputCol);
			std::wostringstream msg;
			msg << "Mismatch at filter " << filter << ", row " << outputRow << ", column " << outputCol;
			Assert::AreEqual(expectedActivation, actual, 1e-5, msg.str().c_str());
		  }
		}
	  }
	}
  };
}
