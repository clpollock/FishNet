#include "stdafx.h"
#include "CppUnitTest.h"
#include "ConvolutionalLayer.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace ConvolutionalBackpropagationTests
{
  TEST_CLASS(ConvolutionalBackpropagationTests)
  {
  public:
	TEST_METHOD(SingleChannelTriple3x3FilterStride1ConvolutionalLayerBackpropagateError)
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

	  double outputErrors[3 * 6 * 6] =
	  {
		-0.96678, -0.78373, -0.55794, 0.61000, 0.15480, 0.54525,
		0.51857, 0.98925, 0.73807, -0.21690, 0.22571, 0.74447,
		-0.84138, 0.46042, 0.72620, -0.06277, 0.83815, 0.71590,
		0.25999, -0.21358, 0.07483, 0.68542, 0.35257, 0.65511,
		-0.31429, 0.49485, 0.37355, -0.72522, -0.05983, -0.85455,
		0.73681, 0.01262, 0.47648, -0.81299, 0.82210, -0.52620,
		0.46272, -0.43281, -0.23503, 0.20874, -0.34779, -0.04097,
		-0.30844, 0.98802, 0.67215, -0.88021, 0.53604, -0.50555,
		-0.13694, 0.79932, -0.34098, -0.17982, 0.27516, 0.85690,
		0.29730, 0.73544, -0.81252, 0.21409, -0.15505, 0.03118,
		-0.91480, -0.90857, -0.18155, 0.64351, -0.24117, -0.07476,
		-0.24550, -0.50893, -0.42020, 0.32371, -0.92890, -0.67828,
		0.70364, -0.67919, -0.03534, 0.22645, -0.26145, -0.07216,
		0.29211, 0.69104, -0.07076, -0.89246, -0.60748, 0.98246,
		0.38547, 0.09226, -0.81086, -0.37228, -0.77526, 0.55240,
		0.20032, -0.83777, 0.74169, 0.58163, -0.74490, 0.11393,
		0.10815, 0.23209, 0.41872, -0.86880, -0.98406, 0.29992,
		-0.93250, -0.14714, -0.05385, 0.28263, -0.77603, 0.16565
	  };

	  double biases[3] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 27), 3, 1, 3, 3);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));
	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 8, 8, 1, 0, nullptr);
	  Tensor inputErrorTensor(1, 8, 8);
	  Tensor outputErrorTensor(std::initializer_list<double>(outputErrors, outputErrors + (3 * 6 * 6)), 3, 6, 6);
	  layer.BackpropagateError(outputErrorTensor, inputErrorTensor, nullptr);

	  Tensor expectedError(8, 8);
	  for (int32_t filter = 0; filter < 3; ++filter)
	  {
		for (int32_t outputRow = 0; outputRow < 6; ++outputRow)
		{
		  for (int32_t outputCol = 0; outputCol < 6; ++outputCol)
		  {
			const double* weight = weights + (filter * 9);
			for (int32_t weightRow = 0; weightRow < 3; ++weightRow)
			{
			  for (int32_t weightCol = 0; weightCol < 3; ++weightCol)
			  {
				double* error = expectedError.ElementAddress(outputRow + weightRow, outputCol + weightCol);
				*error += (*weight * outputErrorTensor.Get(filter, outputRow, outputCol));
				++weight;
			  }
			}
		  }
		}
	  }

	  for (uint32_t inputRow = 0; inputRow < inputErrorTensor.Rows(); ++inputRow)
	  {
		for (uint32_t inputCol = 0; inputCol < inputErrorTensor.Columns(); ++inputCol)
		{
		  std::wostringstream msg;
		  msg << "Mismatch at row " << inputRow << ", column " << inputCol;
		  Assert::AreEqual(expectedError.Get(inputRow, inputCol), inputErrorTensor.Get(0, inputRow, inputCol), 1e-5, msg.str().c_str());
		}
	  }
	}

	TEST_METHOD(SingleChannelTriple3x3FilterStride1ZeroPad1ConvolutionalLayerBackpropagateError)
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

	  double outputErrors[3 * 6 * 6] =
	  {
		-0.96678, -0.78373, -0.55794, 0.61000, 0.15480, 0.54525,
		0.51857, 0.98925, 0.73807, -0.21690, 0.22571, 0.74447,
		-0.84138, 0.46042, 0.72620, -0.06277, 0.83815, 0.71590,
		0.25999, -0.21358, 0.07483, 0.68542, 0.35257, 0.65511,
		-0.31429, 0.49485, 0.37355, -0.72522, -0.05983, -0.85455,
		0.73681, 0.01262, 0.47648, -0.81299, 0.82210, -0.52620,
		0.46272, -0.43281, -0.23503, 0.20874, -0.34779, -0.04097,
		-0.30844, 0.98802, 0.67215, -0.88021, 0.53604, -0.50555,
		-0.13694, 0.79932, -0.34098, -0.17982, 0.27516, 0.85690,
		0.29730, 0.73544, -0.81252, 0.21409, -0.15505, 0.03118,
		-0.91480, -0.90857, -0.18155, 0.64351, -0.24117, -0.07476,
		-0.24550, -0.50893, -0.42020, 0.32371, -0.92890, -0.67828,
		0.70364, -0.67919, -0.03534, 0.22645, -0.26145, -0.07216,
		0.29211, 0.69104, -0.07076, -0.89246, -0.60748, 0.98246,
		0.38547, 0.09226, -0.81086, -0.37228, -0.77526, 0.55240,
		0.20032, -0.83777, 0.74169, 0.58163, -0.74490, 0.11393,
		0.10815, 0.23209, 0.41872, -0.86880, -0.98406, 0.29992,
		-0.93250, -0.14714, -0.05385, 0.28263, -0.77603, 0.16565
	  };

	  double biases[3] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 27), 3, 1, 3, 3);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));
	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 6, 6, 1, 1, nullptr);
	  Tensor inputErrorTensor(1, 6, 6);
	  Tensor outputErrorTensor(std::initializer_list<double>(outputErrors, outputErrors + (3 * 6 * 6)), 3, 6, 6);
	  layer.BackpropagateError(outputErrorTensor, inputErrorTensor, nullptr);

	  Tensor expectedError(6, 6);
	  for (int32_t filter = 0; filter < 3; ++filter)
	  {
		int outputRow = 0;
		for (int32_t inputRow = -1; inputRow < 5; ++inputRow)
		{
		  int outputCol = 0;
		  for (int32_t inputCol = -1; inputCol < 5; ++inputCol)
		  {
			const double* weight = weights + (filter * 9);
			for (int32_t weightRow = 0; weightRow < 3; ++weightRow)
			{
			  for (int32_t weightCol = 0; weightCol < 3; ++weightCol)
			  {
				int32_t row = inputRow + weightRow;
				int32_t col = inputCol + weightCol;
				if (row >= 0 && col >= 0 && row < 6 && col < 6)
				{
				  double* error = expectedError.ElementAddress(row, col);
				  *error += (*weight * outputErrorTensor.Get(filter, outputRow, outputCol));
				}
				++weight;
			  }
			}
			++outputCol;
		  }
		  ++outputRow;
		}
	  }

	  for (uint32_t inputRow = 0; inputRow < inputErrorTensor.Rows(); ++inputRow)
	  {
		for (uint32_t inputCol = 0; inputCol < inputErrorTensor.Columns(); ++inputCol)
		{
		  std::wostringstream msg;
		  msg << "Mismatch at row " << inputRow << ", column " << inputCol;
		  Assert::AreEqual(expectedError.Get(inputRow, inputCol), inputErrorTensor.Get(0, inputRow, inputCol), 1e-5, msg.str().c_str());
		}
	  }
	}

	TEST_METHOD(TwinChannelTriple3x3FilterStride1ConvolutionalLayerBackpropagateError)
	{
	  double weights[54] =
	  {
		0.46423, -0.92972, 0.13349,
		-0.97049, 0.84645, -0.97465,
		0.30291, 0.97435, -0.76644,
		0.49702, 0.33716, 0.08387,
		-0.93769, 0.87674, 0.65471,
		0.44367, -0.31517, 0.21676,
		-0.08736, 0.61972, 0.22116,
		-0.15218, -0.32095, 0.33927,
		0.58528, 0.62483, -0.93393,
		0.64969, -0.13885, -0.97498,
		0.38475, -0.54837, 0.08731,
		0.43720, -0.23487, 0.21276,
		-0.57351, 0.34800, -0.97314,
		0.49121, -0.34730, 0.59259,
		-0.83776, -0.92329, 0.70743,
		0.62497, -0.87396, 0.52005,
		0.48892, 0.55242, 0.22279,
		-0.11647, -0.09981, -0.96230
	  };

	  double outputErrors[3 * 6 * 6] =
	  {
		-0.96678, -0.78373, -0.55794, 0.61000, 0.15480, 0.54525,
		0.51857, 0.98925, 0.73807, -0.21690, 0.22571, 0.74447,
		-0.84138, 0.46042, 0.72620, -0.06277, 0.83815, 0.71590,
		0.25999, -0.21358, 0.07483, 0.68542, 0.35257, 0.65511,
		-0.31429, 0.49485, 0.37355, -0.72522, -0.05983, -0.85455,
		0.73681, 0.01262, 0.47648, -0.81299, 0.82210, -0.52620,
		0.46272, -0.43281, -0.23503, 0.20874, -0.34779, -0.04097,
		-0.30844, 0.98802, 0.67215, -0.88021, 0.53604, -0.50555,
		-0.13694, 0.79932, -0.34098, -0.17982, 0.27516, 0.85690,
		0.29730, 0.73544, -0.81252, 0.21409, -0.15505, 0.03118,
		-0.91480, -0.90857, -0.18155, 0.64351, -0.24117, -0.07476,
		-0.24550, -0.50893, -0.42020, 0.32371, -0.92890, -0.67828,
		0.70364, -0.67919, -0.03534, 0.22645, -0.26145, -0.07216,
		0.29211, 0.69104, -0.07076, -0.89246, -0.60748, 0.98246,
		0.38547, 0.09226, -0.81086, -0.37228, -0.77526, 0.55240,
		0.20032, -0.83777, 0.74169, 0.58163, -0.74490, 0.11393,
		0.10815, 0.23209, 0.41872, -0.86880, -0.98406, 0.29992,
		-0.93250, -0.14714, -0.05385, 0.28263, -0.77603, 0.16565
	  };

	  double biases[3] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 54), 3, 2, 3, 3);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));
	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 8, 8, 1, 0, nullptr);

	  Tensor inputErrorTensor(2, 8, 8);
	  Tensor outputErrorTensor(std::initializer_list<double>(outputErrors, outputErrors + (3 * 6 * 6)), 3, 6, 6);
	  layer.BackpropagateError(outputErrorTensor, inputErrorTensor, nullptr);

	  Tensor expectedError(2, 8, 8);
	  for (int32_t filter = 0; filter < 3; ++filter)
	  {
		for (int32_t inputChannel = 0; inputChannel < 2; ++inputChannel)
		{
		  for (int32_t outputRow = 0; outputRow < 6; ++outputRow)
		  {
			for (int32_t outputCol = 0; outputCol < 6; ++outputCol)
			{
			  const double* weight = weights + (filter * 18) + (inputChannel * 9);
			  for (int32_t weightRow = 0; weightRow < 3; ++weightRow)
			  {
				for (int32_t weightCol = 0; weightCol < 3; ++weightCol)
				{
				  double* error = expectedError.ElementAddress(inputChannel, outputRow + weightRow, outputCol + weightCol);
				  *error += (*weight * outputErrorTensor.Get(filter, outputRow, outputCol));
				  ++weight;
				}
			  }
			}
		  }
		}
	  }

	  for (int32_t inputChannel = 0; inputChannel < 2; ++inputChannel)
	  {
		for (uint32_t inputRow = 0; inputRow < inputErrorTensor.Rows(); ++inputRow)
		{
		  for (uint32_t inputCol = 0; inputCol < inputErrorTensor.Columns(); ++inputCol)
		  {
			std::wostringstream msg;
			msg << "Mismatch at row " << inputRow << ", column " << inputCol;
			Assert::AreEqual(expectedError.Get(inputChannel, inputRow, inputCol), inputErrorTensor.Get(inputChannel, inputRow, inputCol), 1e-5, msg.str().c_str());
		  }
		}
	  }
	}

	TEST_METHOD(TwinChannelTriple3x3FilterStride1ZeroPad1ConvolutionalLayerBackpropagateError)
	{
	  double weights[54] =
	  {
		0.46423, -0.92972, 0.13349,
		-0.97049, 0.84645, -0.97465,
		0.30291, 0.97435, -0.76644,
		0.49702, 0.33716, 0.08387,
		-0.93769, 0.87674, 0.65471,
		0.44367, -0.31517, 0.21676,
		-0.08736, 0.61972, 0.22116,
		-0.15218, -0.32095, 0.33927,
		0.58528, 0.62483, -0.93393,
		0.64969, -0.13885, -0.97498,
		0.38475, -0.54837, 0.08731,
		0.43720, -0.23487, 0.21276,
		-0.57351, 0.34800, -0.97314,
		0.49121, -0.34730, 0.59259,
		-0.83776, -0.92329, 0.70743,
		0.62497, -0.87396, 0.52005,
		0.48892, 0.55242, 0.22279,
		-0.11647, -0.09981, -0.96230
	  };

	  double outputErrors[3 * 6 * 6] =
	  {
		-0.96678, -0.78373, -0.55794, 0.61000, 0.15480, 0.54525,
		0.51857, 0.98925, 0.73807, -0.21690, 0.22571, 0.74447,
		-0.84138, 0.46042, 0.72620, -0.06277, 0.83815, 0.71590,
		0.25999, -0.21358, 0.07483, 0.68542, 0.35257, 0.65511,
		-0.31429, 0.49485, 0.37355, -0.72522, -0.05983, -0.85455,
		0.73681, 0.01262, 0.47648, -0.81299, 0.82210, -0.52620,
		0.46272, -0.43281, -0.23503, 0.20874, -0.34779, -0.04097,
		-0.30844, 0.98802, 0.67215, -0.88021, 0.53604, -0.50555,
		-0.13694, 0.79932, -0.34098, -0.17982, 0.27516, 0.85690,
		0.29730, 0.73544, -0.81252, 0.21409, -0.15505, 0.03118,
		-0.91480, -0.90857, -0.18155, 0.64351, -0.24117, -0.07476,
		-0.24550, -0.50893, -0.42020, 0.32371, -0.92890, -0.67828,
		0.70364, -0.67919, -0.03534, 0.22645, -0.26145, -0.07216,
		0.29211, 0.69104, -0.07076, -0.89246, -0.60748, 0.98246,
		0.38547, 0.09226, -0.81086, -0.37228, -0.77526, 0.55240,
		0.20032, -0.83777, 0.74169, 0.58163, -0.74490, 0.11393,
		0.10815, 0.23209, 0.41872, -0.86880, -0.98406, 0.29992,
		-0.93250, -0.14714, -0.05385, 0.28263, -0.77603, 0.16565
	  };

	  double biases[3] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + 54), 3, 2, 3, 3);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));
	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 6, 6, 1, 1, nullptr);

	  Tensor inputErrorTensor(2, 6, 6);
	  Tensor outputErrorTensor(std::initializer_list<double>(outputErrors, outputErrors + (3 * 6 * 6)), 3, 6, 6);
	  layer.BackpropagateError(outputErrorTensor, inputErrorTensor, nullptr);

	  Tensor expectedError(2, 6, 6);
	  for (int32_t filter = 0; filter < 3; ++filter)
	  {
		for (int32_t inputChannel = 0; inputChannel < 2; ++inputChannel)
		{
		  int outputRow = 0;
		  for (int32_t inputRow = -1; inputRow < 5; ++inputRow)
		  {
			int outputCol = 0;
			for (int32_t inputCol = -1; inputCol < 5; ++inputCol)
			{
			  const double* weight = weights + (filter * 18) + (inputChannel * 9);
			  for (int32_t weightRow = 0; weightRow < 3; ++weightRow)
			  {
				for (int32_t weightCol = 0; weightCol < 3; ++weightCol)
				{
				  int32_t row = inputRow + weightRow;
				  int32_t col = inputCol + weightCol;
				  if (row >= 0 && col >= 0 && row < 6 && col < 6)
				  {
					double* error = expectedError.ElementAddress(inputChannel, row, col);
					*error += (*weight * outputErrorTensor.Get(filter, outputRow, outputCol));
				  }
				  ++weight;
				}
			  }
			  ++outputCol;
			}
			++outputRow;
		  }
		}
	  }

	  for (int32_t inputChannel = 0; inputChannel < 2; ++inputChannel)
	  {
		for (uint32_t inputRow = 0; inputRow < inputErrorTensor.Rows(); ++inputRow)
		{
		  for (uint32_t inputCol = 0; inputCol < inputErrorTensor.Columns(); ++inputCol)
		  {
			std::wostringstream msg;
			msg << "Mismatch at row " << inputRow << ", column " << inputCol;
			Assert::AreEqual(expectedError.Get(inputChannel, inputRow, inputCol), inputErrorTensor.Get(inputChannel, inputRow, inputCol), 1e-5, msg.str().c_str());
		  }
		}
	  }
	}

	TEST_METHOD(TwinChannelTriple4x4FilterStride2ConvolutionalLayerBackpropagateError)
	{
	  double weights[3*2*4*4] =
	  {
		-0.01108, 0.97816, -0.78641, -0.89270,
		-0.33174, 0.30528, -0.04527, 0.35553,
		-0.49700, 0.58814, 0.90204, -0.48834,
		0.65249, -0.80821, 0.57916, -0.04153,
		-0.36100, -0.51677, -0.08581, -0.16559,
		0.99117, 0.47608, 0.39073, -0.29230,
		-0.11487, 0.16987, 0.51307, -0.16525,
		-0.75326, 0.47156, -0.14763, -0.43065,
		0.04710, -0.62656, -0.33185, -0.20619,
		-0.32372, -0.57235, 0.12878, -0.17378,
		0.70062, 0.88081, 0.43133, 0.70231,
		-0.36345, 0.13469, -0.93650, -0.38344,
		0.39351, -0.81123, 0.87519, -0.99541,
		-0.40693, 0.57371, 0.42559, 0.96086,
		-0.04616, 0.54760, -0.85907, 0.64139,
		-0.42528, -0.20924, 0.80186, 0.75696,
		0.20094, -0.01190, 0.85828, -0.59205,
		0.94499, 0.34961, -0.60954, -0.97884,
		-0.40729, 0.91654, -0.79315, -0.41908,
		0.70304, -0.86860, -0.44146, 0.10882,
		0.78825, 0.91760, 0.22370, -0.67696,
		-0.61214, -0.44669, 0.74376, -0.93622,
		0.33902, -0.92020, 0.21658, 0.41969,
		0.22020, -0.34360, -0.21733, -0.26004
	  };

	  double outputErrors[3 * 6 * 6] =
	  {
		-0.96678, -0.78373, -0.55794, 0.61000, 0.15480, 0.54525,
		0.51857, 0.98925, 0.73807, -0.21690, 0.22571, 0.74447,
		-0.84138, 0.46042, 0.72620, -0.06277, 0.83815, 0.71590,
		0.25999, -0.21358, 0.07483, 0.68542, 0.35257, 0.65511,
		-0.31429, 0.49485, 0.37355, -0.72522, -0.05983, -0.85455,
		0.73681, 0.01262, 0.47648, -0.81299, 0.82210, -0.52620,
		0.46272, -0.43281, -0.23503, 0.20874, -0.34779, -0.04097,
		-0.30844, 0.98802, 0.67215, -0.88021, 0.53604, -0.50555,
		-0.13694, 0.79932, -0.34098, -0.17982, 0.27516, 0.85690,
		0.29730, 0.73544, -0.81252, 0.21409, -0.15505, 0.03118,
		-0.91480, -0.90857, -0.18155, 0.64351, -0.24117, -0.07476,
		-0.24550, -0.50893, -0.42020, 0.32371, -0.92890, -0.67828,
		0.70364, -0.67919, -0.03534, 0.22645, -0.26145, -0.07216,
		0.29211, 0.69104, -0.07076, -0.89246, -0.60748, 0.98246,
		0.38547, 0.09226, -0.81086, -0.37228, -0.77526, 0.55240,
		0.20032, -0.83777, 0.74169, 0.58163, -0.74490, 0.11393,
		0.10815, 0.23209, 0.41872, -0.86880, -0.98406, 0.29992,
		-0.93250, -0.14714, -0.05385, 0.28263, -0.77603, 0.16565
	  };

	  double biases[3] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + (3 * 2 * 4 * 4)), 3, 2, 4, 4);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));
	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 14, 14, 2, 0, nullptr);

	  Tensor inputErrorTensor(2, 14, 14);
	  Tensor outputErrorTensor(std::initializer_list<double>(outputErrors, outputErrors + (3 * 6 * 6)), 3, 6, 6);
	  layer.BackpropagateError(outputErrorTensor, inputErrorTensor, nullptr);

	  Tensor expectedError(2, 14, 14);
	  for (int32_t filter = 0; filter < 3; ++filter)
	  {
		for (int32_t inputChannel = 0; inputChannel < 2; ++inputChannel)
		{
		  for (int32_t inputRow = 0; inputRow <= 10; inputRow += 2)
		  {
			for (int32_t inputCol = 0; inputCol <= 10; inputCol += 2)
			{
			  const double* weight = weights + (filter * 2 * 4 * 4) + (inputChannel * 4 * 4);
			  for (int32_t weightRow = 0; weightRow < 4; ++weightRow)
			  {
				for (int32_t weightCol = 0; weightCol < 4; ++weightCol)
				{
				  // We're using a stride of 2, so output row and col will be half the input row and col.
				  double* error = expectedError.ElementAddress(inputChannel, inputRow + weightRow, inputCol + weightCol);
				  *error += (*weight * outputErrorTensor.Get(filter, inputRow / 2, inputCol / 2));
				  ++weight;
				}
			  }
			}
		  }
		}
	  }

	  for (int32_t inputChannel = 0; inputChannel < 2; ++inputChannel)
	  {
		for (uint32_t inputRow = 0; inputRow < expectedError.Rows(); ++inputRow)
		{
		  for (uint32_t inputCol = 0; inputCol < expectedError.Columns(); ++inputCol)
		  {
			std::wostringstream msg;
			msg << "Mismatch at row " << inputRow << ", column " << inputCol;
			Assert::AreEqual(expectedError.Get(inputChannel, inputRow, inputCol), inputErrorTensor.Get(inputChannel, inputRow, inputCol), 1e-5, msg.str().c_str());
		  }
		}
	  }
	}

	TEST_METHOD(TwinChannelTriple4x4FilterStride2ZeroPad2ConvolutionalLayerBackpropagateError)
	{
	  double weights[3*2*4*4] =
	  {
		-0.01108, 0.97816, -0.78641, -0.89270,
		-0.33174, 0.30528, -0.04527, 0.35553,
		-0.49700, 0.58814, 0.90204, -0.48834,
		0.65249, -0.80821, 0.57916, -0.04153,
		-0.36100, -0.51677, -0.08581, -0.16559,
		0.99117, 0.47608, 0.39073, -0.29230,
		-0.11487, 0.16987, 0.51307, -0.16525,
		-0.75326, 0.47156, -0.14763, -0.43065,
		0.04710, -0.62656, -0.33185, -0.20619,
		-0.32372, -0.57235, 0.12878, -0.17378,
		0.70062, 0.88081, 0.43133, 0.70231,
		-0.36345, 0.13469, -0.93650, -0.38344,
		0.39351, -0.81123, 0.87519, -0.99541,
		-0.40693, 0.57371, 0.42559, 0.96086,
		-0.04616, 0.54760, -0.85907, 0.64139,
		-0.42528, -0.20924, 0.80186, 0.75696,
		0.20094, -0.01190, 0.85828, -0.59205,
		0.94499, 0.34961, -0.60954, -0.97884,
		-0.40729, 0.91654, -0.79315, -0.41908,
		0.70304, -0.86860, -0.44146, 0.10882,
		0.78825, 0.91760, 0.22370, -0.67696,
		-0.61214, -0.44669, 0.74376, -0.93622,
		0.33902, -0.92020, 0.21658, 0.41969,
		0.22020, -0.34360, -0.21733, -0.26004
	  };

	  double outputErrors[3 * 6 * 6] =
	  {
		-0.96678, -0.78373, -0.55794, 0.61000, 0.15480, 0.54525,
		0.51857, 0.98925, 0.73807, -0.21690, 0.22571, 0.74447,
		-0.84138, 0.46042, 0.72620, -0.06277, 0.83815, 0.71590,
		0.25999, -0.21358, 0.07483, 0.68542, 0.35257, 0.65511,
		-0.31429, 0.49485, 0.37355, -0.72522, -0.05983, -0.85455,
		0.73681, 0.01262, 0.47648, -0.81299, 0.82210, -0.52620,
		0.46272, -0.43281, -0.23503, 0.20874, -0.34779, -0.04097,
		-0.30844, 0.98802, 0.67215, -0.88021, 0.53604, -0.50555,
		-0.13694, 0.79932, -0.34098, -0.17982, 0.27516, 0.85690,
		0.29730, 0.73544, -0.81252, 0.21409, -0.15505, 0.03118,
		-0.91480, -0.90857, -0.18155, 0.64351, -0.24117, -0.07476,
		-0.24550, -0.50893, -0.42020, 0.32371, -0.92890, -0.67828,
		0.70364, -0.67919, -0.03534, 0.22645, -0.26145, -0.07216,
		0.29211, 0.69104, -0.07076, -0.89246, -0.60748, 0.98246,
		0.38547, 0.09226, -0.81086, -0.37228, -0.77526, 0.55240,
		0.20032, -0.83777, 0.74169, 0.58163, -0.74490, 0.11393,
		0.10815, 0.23209, 0.41872, -0.86880, -0.98406, 0.29992,
		-0.93250, -0.14714, -0.05385, 0.28263, -0.77603, 0.16565
	  };

	  double biases[3] = { 0.1, 0.0, -0.2 };
	  auto weightTensor = std::make_unique<Tensor>(std::initializer_list<double>(weights, weights + (3 * 2 * 4 * 4)), 3, 2, 4, 4);
	  auto biasTensor = std::make_unique<Tensor>(std::initializer_list<double>(biases, biases + 3));
	  ConvolutionalLayer layer(std::move(weightTensor), std::move(biasTensor), 10, 10, 2, 2, nullptr);

	  Tensor inputErrorTensor(2, 10, 10);
	  Tensor outputErrorTensor(std::initializer_list<double>(outputErrors, outputErrors + (3 * 6 * 6)), 3, 6, 6);
	  layer.BackpropagateError(outputErrorTensor, inputErrorTensor, nullptr);

	  Tensor expectedError(2, 10, 10);
	  for (int32_t filter = 0; filter < 3; ++filter)
	  {
		for (int32_t inputChannel = 0; inputChannel < 2; ++inputChannel)
		{
		  int outputRow = 0;
		  for (int32_t inputRow = -2; inputRow < 9; inputRow += 2)
		  {
			int outputCol = 0;
			for (int32_t inputCol = -2; inputCol < 9; inputCol += 2)
			{
			  const double* weight = weights + (filter * 2 * 4 * 4) + (inputChannel * 4 * 4);
			  for (int32_t weightRow = 0; weightRow < 4; ++weightRow)
			  {
				for (int32_t weightCol = 0; weightCol < 4; ++weightCol)
				{
				  int32_t row = inputRow + weightRow;
				  int32_t col = inputCol + weightCol;
				  if (row >= 0 && col >= 0 && row < 10 && col < 10)
				  {
					double* error = expectedError.ElementAddress(inputChannel, row, col);
					*error += (*weight * outputErrorTensor.Get(filter, outputRow, outputCol));
				  }
				  ++weight;
				}
			  }
			  ++outputCol;
			}
			++outputRow;
		  }
		}
	  }

	  for (int32_t inputChannel = 0; inputChannel < 2; ++inputChannel)
	  {
		for (uint32_t inputRow = 0; inputRow < expectedError.Rows(); ++inputRow)
		{
		  for (uint32_t inputCol = 0; inputCol < expectedError.Columns(); ++inputCol)
		  {
			std::wostringstream msg;
			msg << "Mismatch at row " << inputRow << ", column " << inputCol;
			Assert::AreEqual(expectedError.Get(inputChannel, inputRow, inputCol), inputErrorTensor.Get(inputChannel, inputRow, inputCol), 1e-5, msg.str().c_str());
		  }
		}
	  }
	}

	TEST_METHOD(SingleChannelTriple4x4FilterStride1ConvolutionalLayerUpdateWeightAndBiasErrors)
	{
	  double delta[3 * 6 * 6] =
	  {
		-0.96678, -0.78373, -0.55794, 0.61000, 0.15480, 0.54525,
		0.51857, 0.98925, 0.73807, -0.21690, 0.22571, 0.74447,
		-0.84138, 0.46042, 0.72620, -0.06277, 0.83815, 0.71590,
		0.25999, -0.21358, 0.07483, 0.68542, 0.35257, 0.65511,
		-0.31429, 0.49485, 0.37355, -0.72522, -0.05983, -0.85455,
		0.73681, 0.01262, 0.47648, -0.81299, 0.82210, -0.52620,
		0.46272, -0.43281, -0.23503, 0.20874, -0.34779, -0.04097,
		-0.30844, 0.98802, 0.67215, -0.88021, 0.53604, -0.50555,
		-0.13694, 0.79932, -0.34098, -0.17982, 0.27516, 0.85690,
		0.29730, 0.73544, -0.81252, 0.21409, -0.15505, 0.03118,
		-0.91480, -0.90857, -0.18155, 0.64351, -0.24117, -0.07476,
		-0.24550, -0.50893, -0.42020, 0.32371, -0.92890, -0.67828,
		0.70364, -0.67919, -0.03534, 0.22645, -0.26145, -0.07216,
		0.29211, 0.69104, -0.07076, -0.89246, -0.60748, 0.98246,
		0.38547, 0.09226, -0.81086, -0.37228, -0.77526, 0.55240,
		0.20032, -0.83777, 0.74169, 0.58163, -0.74490, 0.11393,
		0.10815, 0.23209, 0.41872, -0.86880, -0.98406, 0.29992,
		-0.93250, -0.14714, -0.05385, 0.28263, -0.77603, 0.16565
	  };
	  Tensor deltaTensor(std::initializer_list<double>(delta, delta + (36 * 3)), 3, 6, 6);
	  double prevActivations[9 * 9] =
	  {
		0.87037, -0.27887, -0.43798, -0.40608, -0.17828, 0.74626, 0.61135, -0.00101, 0.31945,
		0.92409, 0.62275, -0.48405, 0.03635, -0.09436, 0.33511, 0.01229, 0.01950, -0.31475,
		0.06185, 0.57762, -0.06239, -0.44511, -0.44659, 0.46489, 0.89225, 0.25210, -0.45245,
		-0.12005, -0.16700, 0.02537, -0.52532, -0.30569, 0.10160, 0.67040, -0.13788, 0.88144,
		-0.40010, -0.87919, -0.25610, -0.96083, -0.56738, 0.37039, -0.16946, -0.16844, -0.94470,
		0.25452, -0.83606, -0.79022, -0.37777, -0.49692, 0.03976, 0.19388, 0.56742, -0.22505,
		0.30234, -0.78069, -0.39916, 0.80905, 0.45502, -0.28509, -0.31133, -0.40910, 0.57071,
		-0.01049, -0.86779, -0.64236, -0.02799, 0.66569, -0.02236, -0.49110, 0.14429, 0.01785,
		0.03179, 0.89346, 0.14010, 0.36333, -0.04649, 0.73117, -0.62910, -0.01262, 0.50025
	  };
	  Tensor prevActivationsTensor(std::initializer_list<double>(prevActivations, prevActivations + 81), 1, 9, 9);

	  ConvolutionalLayer layer(1, 9, 9, 3, 4, 1, 0, nullptr);
	  layer.InitializeWeights();
	  Tensor nablaW(3, 1, 4, 4);
	  Tensor nablaB(3);
	  layer.UpdateWeightAndBiasErrors(deltaTensor, prevActivationsTensor, nablaW, nablaB, nullptr);

	  for (int filter = 0; filter < 3; ++filter)
	  {
		for (int filterRow = 0; filterRow < 4; ++filterRow)
		{
		  for (int filterCol = 0; filterCol < 4; ++filterCol)
		  {
			double expectedWeightError = 0.0;
			for (int deltaRow = 0; deltaRow < 6; ++deltaRow)
			{
			  for (int deltaCol = 0; deltaCol < 6; ++deltaCol)
			  {
				double del = delta[filter * 36 + deltaRow * 6 + deltaCol];
				double activation = prevActivations[(deltaRow + filterRow) * 9 + deltaCol  + filterCol];
				expectedWeightError += (del * activation);
			  }
			}
			std::wostringstream msg;
			msg << "NablaW mismatch at filter " << filter << ", row " << filterRow << ", column " << filterCol;
			Assert::AreEqual(expectedWeightError, nablaW.Get(filter, 0, filterRow, filterCol), 1e-5, msg.str().c_str());
		  }
		}

		double expectedBiasError = 0.0;
		for (int deltaRow = 0; deltaRow < 6; ++deltaRow)
		{
		  for (int deltaCol = 0; deltaCol < 6; ++deltaCol)
			expectedBiasError += delta[filter * 36 + deltaRow * 6 + deltaCol];
		}
		std::wostringstream msg;
		msg << "NablaB mismatch at filter " << filter;
		Assert::AreEqual(expectedBiasError, nablaB.Get(filter), 1e-5, msg.str().c_str());
	  }
	}

	TEST_METHOD(SingleChannelTriple4x4FilterStride1ZeroPad2ConvolutionalLayerUpdateWeightAndBiasErrors)
	{
	  double delta[3 * 6 * 6] =
	  {
		-0.96678, -0.78373, -0.55794, 0.61000, 0.15480, 0.54525,
		0.51857, 0.98925, 0.73807, -0.21690, 0.22571, 0.74447,
		-0.84138, 0.46042, 0.72620, -0.06277, 0.83815, 0.71590,
		0.25999, -0.21358, 0.07483, 0.68542, 0.35257, 0.65511,
		-0.31429, 0.49485, 0.37355, -0.72522, -0.05983, -0.85455,
		0.73681, 0.01262, 0.47648, -0.81299, 0.82210, -0.52620,
		0.46272, -0.43281, -0.23503, 0.20874, -0.34779, -0.04097,
		-0.30844, 0.98802, 0.67215, -0.88021, 0.53604, -0.50555,
		-0.13694, 0.79932, -0.34098, -0.17982, 0.27516, 0.85690,
		0.29730, 0.73544, -0.81252, 0.21409, -0.15505, 0.03118,
		-0.91480, -0.90857, -0.18155, 0.64351, -0.24117, -0.07476,
		-0.24550, -0.50893, -0.42020, 0.32371, -0.92890, -0.67828,
		0.70364, -0.67919, -0.03534, 0.22645, -0.26145, -0.07216,
		0.29211, 0.69104, -0.07076, -0.89246, -0.60748, 0.98246,
		0.38547, 0.09226, -0.81086, -0.37228, -0.77526, 0.55240,
		0.20032, -0.83777, 0.74169, 0.58163, -0.74490, 0.11393,
		0.10815, 0.23209, 0.41872, -0.86880, -0.98406, 0.29992,
		-0.93250, -0.14714, -0.05385, 0.28263, -0.77603, 0.16565
	  };
	  Tensor deltaTensor(std::initializer_list<double>(delta, delta + (36 * 3)), 3, 6, 6);
	  double prevActivations[5 * 5] =
	  {
		0.06185, 0.57762, -0.06239, -0.44511, -0.44659,
		-0.12005, -0.16700, 0.02537, -0.52532, -0.30569,
		-0.40010, -0.87919, -0.25610, -0.96083, -0.56738,
		0.25452, -0.83606, -0.79022, -0.37777, -0.49692,
		0.30234, -0.78069, -0.39916, 0.80905, 0.45502
	  };
	  Tensor prevActivationsTensor(std::initializer_list<double>(prevActivations, prevActivations + 25), 1, 5, 5);

	  ConvolutionalLayer layer(1, 5, 5, 3, 4, 1, 2, nullptr);
	  layer.InitializeWeights();
	  Tensor nablaW(3, 1, 4, 4);
	  Tensor nablaB(3);
	  layer.UpdateWeightAndBiasErrors(deltaTensor, prevActivationsTensor, nablaW, nablaB, nullptr);

	  //
	  for (int filter = 0; filter < 3; ++filter)
	  {
		for (int filterRow = 0; filterRow < 4; ++filterRow)
		{
		  for (int filterCol = 0; filterCol < 4; ++filterCol)
		  {
			double expectedWeightError = 0.0;
			for (int deltaRow = 0; deltaRow < 6; ++deltaRow)
			{
			  for (int deltaCol = 0; deltaCol < 6; ++deltaCol)
			  {
				double del = delta[filter * 36 + deltaRow * 6 + deltaCol];
				int prevRow = deltaRow - 2 + filterRow;
				int prevCol = deltaCol - 2 + filterCol;
				if (prevRow >= 0 && prevRow < 5 && prevCol >= 0 && prevCol < 5)
				{
				  double activation = prevActivations[prevRow * 5 + prevCol];
				  expectedWeightError += (del * activation);
				}
			  }
			}
			std::wostringstream msg;
			msg << "NablaW mismatch at filter " << filter << ", row " << filterRow << ", column " << filterCol;
			Assert::AreEqual(expectedWeightError, nablaW.Get(filter, 0, filterRow, filterCol), 1e-5, msg.str().c_str());
		  }
		}

		double expectedBiasError = 0.0;
		for (int deltaRow = 0; deltaRow < 6; ++deltaRow)
		{
		  for (int deltaCol = 0; deltaCol < 6; ++deltaCol)
			expectedBiasError += delta[filter * 36 + deltaRow * 6 + deltaCol];
		}
		std::wostringstream msg;
		msg << "NablaB mismatch at filter " << filter;
		Assert::AreEqual(expectedBiasError, nablaB.Get(filter), 1e-5, msg.str().c_str());
	  }
	}

	TEST_METHOD(FourChannelTriple4x4FilterStride1ConvolutionalLayerUpdateWeightAndBiasErrors)
	{
	  double delta[3 * 6 * 6] =
	  {
		-0.96678, -0.78373, -0.55794, 0.61000, 0.15480, 0.54525,
		0.51857, 0.98925, 0.73807, -0.21690, 0.22571, 0.74447,
		-0.84138, 0.46042, 0.72620, -0.06277, 0.83815, 0.71590,
		0.25999, -0.21358, 0.07483, 0.68542, 0.35257, 0.65511,
		-0.31429, 0.49485, 0.37355, -0.72522, -0.05983, -0.85455,
		0.73681, 0.01262, 0.47648, -0.81299, 0.82210, -0.52620,
		0.46272, -0.43281, -0.23503, 0.20874, -0.34779, -0.04097,
		-0.30844, 0.98802, 0.67215, -0.88021, 0.53604, -0.50555,
		-0.13694, 0.79932, -0.34098, -0.17982, 0.27516, 0.85690,
		0.29730, 0.73544, -0.81252, 0.21409, -0.15505, 0.03118,
		-0.91480, -0.90857, -0.18155, 0.64351, -0.24117, -0.07476,
		-0.24550, -0.50893, -0.42020, 0.32371, -0.92890, -0.67828,
		0.70364, -0.67919, -0.03534, 0.22645, -0.26145, -0.07216,
		0.29211, 0.69104, -0.07076, -0.89246, -0.60748, 0.98246,
		0.38547, 0.09226, -0.81086, -0.37228, -0.77526, 0.55240,
		0.20032, -0.83777, 0.74169, 0.58163, -0.74490, 0.11393,
		0.10815, 0.23209, 0.41872, -0.86880, -0.98406, 0.29992,
		-0.93250, -0.14714, -0.05385, 0.28263, -0.77603, 0.16565
	  };
	  Tensor deltaTensor(std::initializer_list<double>(delta, delta + (36 * 3)), 3, 6, 6);
	  double prevActivations[4 * 9 * 9] =
	  {
		-0.33818, -0.03791, -0.38952, -0.38243, -0.56507, 0.36559, -0.35899, -0.27600, 0.05529,
		0.10931, -0.94946, -0.64838, 0.79637, 0.71117, 0.11148, -0.00445, 0.37948, 0.90025,
		0.62505, -0.44365, -0.17203, -0.35598, 0.43221, -0.44774, 0.56632, -0.77168, 0.19406,
		0.81186, -0.83588, -0.56024, 0.21459, -0.41484, -0.07461, -0.94417, -0.28229, -0.43900,
		0.90557, -0.04839, -0.46031, -0.66978, 0.37223, 0.77734, -0.54320, -0.56026, -0.83502,
		0.34759, -0.45779, 0.05624, 0.17175, 0.95325, 0.49548, 0.48723, 0.46388, 0.94036,
		-0.73179, 0.40213, 0.15014, -0.40278, 0.47441, -0.60995, -0.59743, -0.92055, 0.02020,
		0.39829, -0.80369, 0.29063, 0.51835, 0.92602, -0.22794, -0.15959, -0.59940, 0.77525,
		0.16861, 0.14238, -0.36656, 0.60456, -0.05017, -0.46567, -0.54005, -0.03325, -0.25899,
		-0.90399, 0.52897, -0.07747, 0.37172, -0.21245, -0.81733, 0.07587, -0.64827, 0.26873,
		0.33611, -0.41342, 0.67896, 0.16032, -0.48612, 0.16966, 0.09471, 0.44696, -0.58354,
		0.69106, -0.09681, 0.23668, -0.31201, 0.13491, -0.06765, 0.33648, -0.68274, -0.42746,
		-0.10415, -0.61627, -0.19456, -0.36614, -0.94196, 0.91276, -0.16243, -0.95317, 0.67269,
		0.59159, -0.57499, 0.27150, 0.10668, -0.59564, 0.46644, 0.46171, 0.24582, -0.78617,
		-0.55303, -0.51030, 0.57432, -0.65156, -0.34578, -0.81397, 0.70866, 0.52199, 0.53141,
		0.32573, -0.05459, -0.79745, -0.26688, 0.73741, 0.25041, -0.93808, -0.52393, -0.54982,
		0.52317, -0.15901, -0.41513, 0.63699, -0.39244, -0.08485, -0.07530, -0.85250, 0.07386,
		-0.71243, 0.37299, -0.56435, -0.55147, 0.56429, 0.59957, -0.08682, -0.60706, 0.02007,
		-0.37949, 0.46795, -0.98657, 0.85911, 0.71236, -0.55384, 0.03887, 0.79121, 0.65091,
		0.06849, -0.29096, 0.41458, 0.91817, -0.10302, -0.22672, -0.20968, 0.79986, -0.46190,
		0.61194, -0.97058, -0.50139, 0.42106, -0.35197, 0.36200, 0.67231, -0.49566, -0.78037,
		-0.43951, 0.27664, 0.82793, -0.36162, -0.25591, 0.70978, 0.11406, 0.08985, -0.59587,
		-0.77758, 0.66323, -0.87170, -0.67247, 0.65863, 0.70973, 0.33918, 0.77690, -0.25736,
		-0.86446, -0.81277, -0.31220, 0.81910, 0.05925, 0.67650, 0.26090, 0.04913, -0.39661,
		-0.94994, 0.50386, -0.57323, -0.06200, 0.20186, -0.87613, -0.28943, -0.04738, -0.40505,
		0.62478, -0.96724, -0.39057, -0.10762, -0.53184, 0.44155, 0.54417, 0.48226, -0.09699,
		-0.53545, 0.66465, 0.84454, 0.64909, 0.16745, -0.62449, -0.72472, 0.94994, 0.83158,
		0.41697, 0.43157, -0.46510, -0.72811, -0.59386, -0.30230, 0.21186, -0.05906, -0.50490,
		0.40354, 0.39391, -0.92061, 0.91489, 0.17669, 0.18401, 0.35135, -0.50456, 0.97620,
		0.36662, -0.02316, 0.46987, 0.08442, 0.85585, 0.35599, 0.86486, -0.08526, -0.72231,
		0.31904, -0.38156, -0.87339, 0.84394, -0.14888, -0.58345, -0.95668, -0.40398, -0.04849,
		-0.36884, 0.46857, 0.32901, 0.73460, -0.71456, 0.45807, -0.24734, -0.63388, -0.78477,
		-0.07160, 0.18846, 0.71003, 0.24907, 0.77256, -0.99770, 0.82716, -0.25968, 0.91256,
		-0.46031, 0.87852, 0.98466, -0.04602, -0.77216, -0.64188, -0.48112, 0.94108, -0.43474,
		-0.34819, -0.28431, 0.85570, 0.18073, 0.95753, -0.78501, 0.67553, 0.55555, 0.40909,
		-0.04552, -0.03513, -0.24786, -0.09754, 0.83445, -0.37437, -0.36114, -0.57687, 0.26398
	  };
	  Tensor prevActivationsTensor(std::initializer_list<double>(prevActivations, prevActivations + 4 * 9 * 9), 4, 9, 9);

	  ConvolutionalLayer layer(4, 9, 9, 3, 4, 1, 0, nullptr);
	  layer.InitializeWeights();
	  Tensor nablaW(3, 4, 4, 4);
	  Tensor nablaB(3);
	  layer.UpdateWeightAndBiasErrors(deltaTensor, prevActivationsTensor, nablaW, nablaB, nullptr);

	  for (int filter = 0; filter < 3; ++filter)
	  {
		for (int channel = 0; channel < 4; ++channel)
		{
		  for (int filterRow = 0; filterRow < 4; ++filterRow)
		  {
			for (int filterCol = 0; filterCol < 4; ++filterCol)
			{
			  double expectedWeightError = 0.0;
			  double* del = delta + (filter * 36);
			  for (int deltaRow = 0; deltaRow < 6; ++deltaRow)
			  {
				for (int deltaCol = 0; deltaCol < 6; ++deltaCol)
				{
				  double activation = prevActivations[(channel * 9 * 9) + ((deltaRow + filterRow) * 9) + deltaCol + filterCol];
				  expectedWeightError += (*del * activation);
				  ++del;
				}
			  }
			  std::wostringstream msg;
			  msg << "NablaW mismatch at filter " << filter << ", row " << filterRow << ", column " << filterCol;
			  Assert::AreEqual(expectedWeightError, nablaW.Get(filter, channel, filterRow, filterCol), 1e-5, msg.str().c_str());
			}
		  }
		}

		double expectedBiasError = 0.0;
		for (int deltaRow = 0; deltaRow < 6; ++deltaRow)
		{
		  for (int deltaCol = 0; deltaCol < 6; ++deltaCol)
			expectedBiasError += delta[filter * 36 + deltaRow * 6 + deltaCol];
		}
		std::wostringstream msg;
		msg << "NablaB mismatch at filter " << filter;
		Assert::AreEqual(expectedBiasError, nablaB.Get(filter), 1e-5, msg.str().c_str());
	  }
	}

	TEST_METHOD(FourChannelTriple4x4FilterStride1ZeroPad2ConvolutionalLayerUpdateWeightAndBiasErrors)
	{
	  double delta[3 * 6 * 6] =
	  {
		-0.96678, -0.78373, -0.55794, 0.61000, 0.15480, 0.54525,
		0.51857, 0.98925, 0.73807, -0.21690, 0.22571, 0.74447,
		-0.84138, 0.46042, 0.72620, -0.06277, 0.83815, 0.71590,
		0.25999, -0.21358, 0.07483, 0.68542, 0.35257, 0.65511,
		-0.31429, 0.49485, 0.37355, -0.72522, -0.05983, -0.85455,
		0.73681, 0.01262, 0.47648, -0.81299, 0.82210, -0.52620,
		0.46272, -0.43281, -0.23503, 0.20874, -0.34779, -0.04097,
		-0.30844, 0.98802, 0.67215, -0.88021, 0.53604, -0.50555,
		-0.13694, 0.79932, -0.34098, -0.17982, 0.27516, 0.85690,
		0.29730, 0.73544, -0.81252, 0.21409, -0.15505, 0.03118,
		-0.91480, -0.90857, -0.18155, 0.64351, -0.24117, -0.07476,
		-0.24550, -0.50893, -0.42020, 0.32371, -0.92890, -0.67828,
		0.70364, -0.67919, -0.03534, 0.22645, -0.26145, -0.07216,
		0.29211, 0.69104, -0.07076, -0.89246, -0.60748, 0.98246,
		0.38547, 0.09226, -0.81086, -0.37228, -0.77526, 0.55240,
		0.20032, -0.83777, 0.74169, 0.58163, -0.74490, 0.11393,
		0.10815, 0.23209, 0.41872, -0.86880, -0.98406, 0.29992,
		-0.93250, -0.14714, -0.05385, 0.28263, -0.77603, 0.16565
	  };
	  Tensor deltaTensor(std::initializer_list<double>(delta, delta + (36 * 3)), 3, 6, 6);
	  double prevActivations[4 * 5 * 5] =
	  {
		0.52317, -0.15901, -0.41513, 0.63699, -0.39244,
		-0.71243, 0.37299, -0.56435, -0.55147, 0.56429,
		-0.37949, 0.46795, -0.98657, 0.85911, 0.71236,
		0.06849, -0.29096, 0.41458, 0.91817, -0.10302,
		0.61194, -0.97058, -0.50139, 0.42106, -0.35197,
		-0.43951, 0.27664, 0.82793, -0.36162, -0.25591,
		-0.77758, 0.66323, -0.87170, -0.67247, 0.65863,
		-0.86446, -0.81277, -0.31220, 0.81910, 0.05925,
		-0.94994, 0.50386, -0.57323, -0.06200, 0.20186,
		0.62478, -0.96724, -0.39057, -0.10762, -0.53184,
		-0.53545, 0.66465, 0.84454, 0.64909, 0.16745,
		0.41697, 0.43157, -0.46510, -0.72811, -0.59386,
		0.40354, 0.39391, -0.92061, 0.91489, 0.17669,
		0.36662, -0.02316, 0.46987, 0.08442, 0.85585,
		0.31904, -0.38156, -0.87339, 0.84394, -0.14888,
		-0.36884, 0.46857, 0.32901, 0.73460, -0.71456,
		-0.07160, 0.18846, 0.71003, 0.24907, 0.77256,
		-0.46031, 0.87852, 0.98466, -0.04602, -0.77216,
		-0.34819, -0.28431, 0.85570, 0.18073, 0.95753,
		-0.04552, -0.03513, -0.24786, -0.09754, 0.83445
	  };
	  Tensor prevActivationsTensor(std::initializer_list<double>(prevActivations, prevActivations + 4 * 5 * 5), 4, 5, 5);

	  ConvolutionalLayer layer(4, 5, 5, 3, 4, 1, 2, nullptr);
	  layer.InitializeWeights();
	  Tensor nablaW(3, 4, 4, 4);
	  Tensor nablaB(3);
	  layer.UpdateWeightAndBiasErrors(deltaTensor, prevActivationsTensor, nablaW, nablaB, nullptr);

	  for (int filter = 0; filter < 3; ++filter)
	  {
		for (int channel = 0; channel < 4; ++channel)
		{
		  for (int filterRow = 0; filterRow < 4; ++filterRow)
		  {
			for (int filterCol = 0; filterCol < 4; ++filterCol)
			{
			  double expectedWeightError = 0.0;
			  double* del = delta + (filter * 36);
			  for (int filterRowOffset = -2; filterRowOffset < 4; ++filterRowOffset)
			  {
				int prevRow = filterRowOffset + filterRow;
				for (int filterColOffset = -2; filterColOffset < 4; ++filterColOffset)
				{
				  int prevCol = filterColOffset + filterCol;
				  if (prevRow >= 0 && prevRow < 5 && prevCol >= 0 && prevCol < 5)
				  {
					double activation = prevActivations[(channel * 5 * 5) + (filterRowOffset + filterRow) * 5 + filterColOffset + filterCol];
					expectedWeightError += (*del * activation);
				  }
				  ++del;
				}
			  }
			  std::wostringstream msg;
			  msg << "NablaW mismatch at filter " << filter << ", row " << filterRow << ", column " << filterCol;
			  Assert::AreEqual(expectedWeightError, nablaW.Get(filter, channel, filterRow, filterCol), 1e-5, msg.str().c_str());
			}
		  }
		}

		double expectedBiasError = 0.0;
		for (int deltaRow = 0; deltaRow < 6; ++deltaRow)
		{
		  for (int deltaCol = 0; deltaCol < 6; ++deltaCol)
			expectedBiasError += delta[filter * 36 + deltaRow * 6 + deltaCol];
		}
		std::wostringstream msg;
		msg << "NablaB mismatch at filter " << filter;
		Assert::AreEqual(expectedBiasError, nablaB.Get(filter), 1e-5, msg.str().c_str());
	  }
	}

	TEST_METHOD(FourChannelTriple3x3FilterStride2ConvolutionalLayerUpdateWeightAndBiasErrors)
	{
	  double delta[3 * 4 * 4] =
	  {
		0.12607, -0.72235, -0.47865, 0.03274,
		0.69829, 0.22116, 0.55568, -0.98420,
		-0.86539, -0.55813, 0.39248, 0.03969,
		0.94210, 0.70328, 0.57449, 0.45554,
		-0.93831, 0.99044, 0.03875, -0.47329,
		-0.77945, 0.56086, -0.60250, -0.88574,
		-0.04890, 0.10612, 0.51434, -0.85138,
		0.50008, 0.76882, 0.09435, -0.99368,
		-0.96563, 0.80097, -0.20009, 0.16932,
		-0.11519, -0.83567, 0.40479, 0.03916,
		0.85537, 0.50877, -0.33722, 0.20106,
		-0.59736, 0.55277, -0.66022, 0.09966
	  };
	  Tensor deltaTensor(std::initializer_list<double>(delta, delta + (3 * 16)), 3, 4, 4);
	  double prevActivations[4 * 9 * 9] =
	  {
		-0.33818, -0.03791, -0.38952, -0.38243, -0.56507, 0.36559, -0.35899, -0.27600, 0.05529,
		0.10931, -0.94946, -0.64838, 0.79637, 0.71117, 0.11148, -0.00445, 0.37948, 0.90025,
		0.62505, -0.44365, -0.17203, -0.35598, 0.43221, -0.44774, 0.56632, -0.77168, 0.19406,
		0.81186, -0.83588, -0.56024, 0.21459, -0.41484, -0.07461, -0.94417, -0.28229, -0.43900,
		0.90557, -0.04839, -0.46031, -0.66978, 0.37223, 0.77734, -0.54320, -0.56026, -0.83502,
		0.34759, -0.45779, 0.05624, 0.17175, 0.95325, 0.49548, 0.48723, 0.46388, 0.94036,
		-0.73179, 0.40213, 0.15014, -0.40278, 0.47441, -0.60995, -0.59743, -0.92055, 0.02020,
		0.39829, -0.80369, 0.29063, 0.51835, 0.92602, -0.22794, -0.15959, -0.59940, 0.77525,
		0.16861, 0.14238, -0.36656, 0.60456, -0.05017, -0.46567, -0.54005, -0.03325, -0.25899,
		-0.90399, 0.52897, -0.07747, 0.37172, -0.21245, -0.81733, 0.07587, -0.64827, 0.26873,
		0.33611, -0.41342, 0.67896, 0.16032, -0.48612, 0.16966, 0.09471, 0.44696, -0.58354,
		0.69106, -0.09681, 0.23668, -0.31201, 0.13491, -0.06765, 0.33648, -0.68274, -0.42746,
		-0.10415, -0.61627, -0.19456, -0.36614, -0.94196, 0.91276, -0.16243, -0.95317, 0.67269,
		0.59159, -0.57499, 0.27150, 0.10668, -0.59564, 0.46644, 0.46171, 0.24582, -0.78617,
		-0.55303, -0.51030, 0.57432, -0.65156, -0.34578, -0.81397, 0.70866, 0.52199, 0.53141,
		0.32573, -0.05459, -0.79745, -0.26688, 0.73741, 0.25041, -0.93808, -0.52393, -0.54982,
		0.52317, -0.15901, -0.41513, 0.63699, -0.39244, -0.08485, -0.07530, -0.85250, 0.07386,
		-0.71243, 0.37299, -0.56435, -0.55147, 0.56429, 0.59957, -0.08682, -0.60706, 0.02007,
		-0.37949, 0.46795, -0.98657, 0.85911, 0.71236, -0.55384, 0.03887, 0.79121, 0.65091,
		0.06849, -0.29096, 0.41458, 0.91817, -0.10302, -0.22672, -0.20968, 0.79986, -0.46190,
		0.61194, -0.97058, -0.50139, 0.42106, -0.35197, 0.36200, 0.67231, -0.49566, -0.78037,
		-0.43951, 0.27664, 0.82793, -0.36162, -0.25591, 0.70978, 0.11406, 0.08985, -0.59587,
		-0.77758, 0.66323, -0.87170, -0.67247, 0.65863, 0.70973, 0.33918, 0.77690, -0.25736,
		-0.86446, -0.81277, -0.31220, 0.81910, 0.05925, 0.67650, 0.26090, 0.04913, -0.39661,
		-0.94994, 0.50386, -0.57323, -0.06200, 0.20186, -0.87613, -0.28943, -0.04738, -0.40505,
		0.62478, -0.96724, -0.39057, -0.10762, -0.53184, 0.44155, 0.54417, 0.48226, -0.09699,
		-0.53545, 0.66465, 0.84454, 0.64909, 0.16745, -0.62449, -0.72472, 0.94994, 0.83158,
		0.41697, 0.43157, -0.46510, -0.72811, -0.59386, -0.30230, 0.21186, -0.05906, -0.50490,
		0.40354, 0.39391, -0.92061, 0.91489, 0.17669, 0.18401, 0.35135, -0.50456, 0.97620,
		0.36662, -0.02316, 0.46987, 0.08442, 0.85585, 0.35599, 0.86486, -0.08526, -0.72231,
		0.31904, -0.38156, -0.87339, 0.84394, -0.14888, -0.58345, -0.95668, -0.40398, -0.04849,
		-0.36884, 0.46857, 0.32901, 0.73460, -0.71456, 0.45807, -0.24734, -0.63388, -0.78477,
		-0.07160, 0.18846, 0.71003, 0.24907, 0.77256, -0.99770, 0.82716, -0.25968, 0.91256,
		-0.46031, 0.87852, 0.98466, -0.04602, -0.77216, -0.64188, -0.48112, 0.94108, -0.43474,
		-0.34819, -0.28431, 0.85570, 0.18073, 0.95753, -0.78501, 0.67553, 0.55555, 0.40909,
		-0.04552, -0.03513, -0.24786, -0.09754, 0.83445, -0.37437, -0.36114, -0.57687, 0.26398
	  };
	  Tensor prevActivationsTensor(std::initializer_list<double>(prevActivations, prevActivations + 4 * 9 * 9), 4, 9, 9);

	  ConvolutionalLayer layer(4, 9, 9, 3, 3, 2, 0, nullptr);
	  layer.InitializeWeights();
	  Tensor nablaW(3, 4, 3, 3);
	  Tensor nablaB(3);
	  layer.UpdateWeightAndBiasErrors(deltaTensor, prevActivationsTensor, nablaW, nablaB, nullptr);

	  for (int filter = 0; filter < 3; ++filter)
	  {
		for (int channel = 0; channel < 4; ++channel)
		{
		  for (int filterRow = 0; filterRow < 3; ++filterRow)
		  {
			for (int filterCol = 0; filterCol < 3; ++filterCol)
			{
			  double expectedWeightError = 0.0;
			  double* del = delta + (filter * 16);
			  for (int deltaRow = 0; deltaRow < 4; ++deltaRow)
			  {
				for (int deltaCol = 0; deltaCol < 4; ++deltaCol)
				{
				  uint32_t inputRow = deltaRow * 2 + filterRow;
				  uint32_t inputCol = deltaCol * 2 + filterCol;
				  double activation = prevActivations[(channel * 9 * 9) + (inputRow * 9) + inputCol];
				  expectedWeightError += (*del * activation);
				  ++del;
				}
			  }
			  std::wostringstream msg;
			  msg << "NablaW mismatch at filter " << filter << ", row " << filterRow << ", column " << filterCol;
			  Assert::AreEqual(expectedWeightError, nablaW.Get(filter, channel, filterRow, filterCol), 1e-5, msg.str().c_str());
			}
		  }
		}

		double expectedBiasError = 0.0;
		for (int deltaRow = 0; deltaRow < 4; ++deltaRow)
		{
		  for (int deltaCol = 0; deltaCol < 4; ++deltaCol)
			expectedBiasError += delta[filter * 16 + (deltaRow * 4) + deltaCol];
		}
		std::wostringstream msg;
		msg << "NablaB mismatch at filter " << filter;
		Assert::AreEqual(expectedBiasError, nablaB.Get(filter), 1e-5, msg.str().c_str());
	  }
	}

	TEST_METHOD(FourChannelTriple3x3FilterStride2ZeroPad1ConvolutionalLayerUpdateWeightAndBiasErrors)
	{
	  double delta[3 * 4 * 4] =
	  {
		0.12607, -0.72235, -0.47865, 0.03274,
		0.69829, 0.22116, 0.55568, -0.98420,
		-0.86539, -0.55813, 0.39248, 0.03969,
		0.94210, 0.70328, 0.57449, 0.45554,
		-0.93831, 0.99044, 0.03875, -0.47329,
		-0.77945, 0.56086, -0.60250, -0.88574,
		-0.04890, 0.10612, 0.51434, -0.85138,
		0.50008, 0.76882, 0.09435, -0.99368,
		-0.96563, 0.80097, -0.20009, 0.16932,
		-0.11519, -0.83567, 0.40479, 0.03916,
		0.85537, 0.50877, -0.33722, 0.20106,
		-0.59736, 0.55277, -0.66022, 0.09966
	  };
	  Tensor deltaTensor(std::initializer_list<double>(delta, delta + (3 * 16)), 3, 4, 4);
	  double prevActivations[4 * 7 * 7] =
	  {
		0.34759, -0.45779, 0.05624, 0.17175, 0.95325, 0.49548, 0.48723,
		0.15014, -0.40278, 0.47441, -0.60995, -0.59743, -0.92055, 0.02020,
		0.39829, -0.80369, 0.29063, 0.51835, 0.92602, -0.22794, -0.15959,
		0.16861, 0.14238, -0.36656, 0.60456, -0.05017, -0.46567, -0.54005,
		-0.07747, 0.37172, -0.21245, -0.81733, 0.07587, -0.64827, 0.26873,
		-0.41342, 0.67896, 0.16032, -0.48612, 0.16966, 0.09471, 0.44696,
		0.69106, -0.31201, 0.13491, -0.06765, 0.33648, -0.68274, -0.42746,
		-0.10415, -0.61627, -0.19456, -0.36614, -0.94196, 0.91276, -0.16243,
		0.59159, 0.10668, -0.59564, 0.46644, 0.46171, 0.24582, -0.78617,
		-0.55303, -0.51030, 0.57432, -0.81397, 0.70866, 0.52199, 0.53141,
		0.32573, -0.05459, -0.79745, 0.25041, -0.93808, -0.52393, -0.54982,
		0.52317, -0.15901, -0.39244, -0.08485, -0.07530, -0.85250, 0.07386,
		-0.71243, 0.37299, -0.56435, -0.55147, 0.56429, 0.59957, 0.02007,
		0.06849, -0.29096, 0.41458, 0.91817, -0.10302, 0.79986, -0.46190,
		0.61194, -0.97058, -0.50139, 0.42106, -0.35197, -0.49566, -0.78037,
		-0.77758, 0.66323, -0.87170, 0.70973, 0.33918, 0.77690, -0.25736,
		-0.86446, 0.81910, 0.05925, 0.67650, 0.26090, 0.04913, -0.39661,
		-0.94994, 0.50386, -0.57323, -0.06200, 0.20186, -0.87613, -0.28943,
		0.62478, -0.10762, -0.53184, 0.44155, 0.54417, 0.48226, -0.09699,
		0.41697, 0.43157, -0.59386, -0.30230, 0.21186, -0.05906, -0.50490,
		0.40354, 0.91489, 0.17669, 0.18401, 0.35135, -0.50456, 0.97620,
		0.36662, -0.02316, 0.46987, 0.08442, 0.86486, -0.08526, -0.72231,
		0.31904, -0.38156, -0.87339, 0.84394, -0.95668, -0.40398, -0.04849,
		-0.36884, 0.46857, 0.32901, 0.73460, -0.71456, 0.45807, -0.24734,
		-0.07160, 0.24907, 0.77256, -0.99770, 0.82716, -0.25968, 0.91256,
		-0.46031, -0.04602, -0.77216, -0.64188, -0.48112, 0.94108, -0.43474,
		-0.34819, -0.28431, 0.85570, 0.18073, 0.95753, 0.55555, 0.40909,
		-0.24786, -0.09754, 0.83445, -0.37437, -0.36114, -0.57687, 0.26398
	  };
	  Tensor prevActivationsTensor(std::initializer_list<double>(prevActivations, prevActivations + 4 * 7 * 7), 4, 7, 7);

	  ConvolutionalLayer layer(4, 7, 7, 3, 3, 2, 1, nullptr);
	  layer.InitializeWeights();
	  Tensor nablaW(3, 4, 3, 3);
	  Tensor nablaB(3);
	  layer.UpdateWeightAndBiasErrors(deltaTensor, prevActivationsTensor, nablaW, nablaB, nullptr);

	  for (int filter = 0; filter < 3; ++filter)
	  {
		for (int channel = 0; channel < 4; ++channel)
		{
		  for (int filterRow = 0; filterRow < 3; ++filterRow)
		  {
			for (int filterCol = 0; filterCol < 3; ++filterCol)
			{
			  double expectedWeightError = 0.0;
			  double* del = delta + (filter * 16);
			  for (int filterRowOffset = -1; filterRowOffset <= 5; filterRowOffset += 2)
			  {
				int prevRow = filterRowOffset + filterRow;
				for (int filterColOffset = -1; filterColOffset <= 5; filterColOffset += 2)
				{
				  int prevCol = filterColOffset + filterCol;
				  if (prevRow >= 0 && prevRow < 7 && prevCol >= 0 && prevCol < 7)
				  {
					double activation = prevActivations[(channel * 7 * 7) + prevRow * 7 + prevCol];
					expectedWeightError += (*del * activation);
				  }
				  ++del;
				}
			  }

			  std::wostringstream msg;
			  msg << "NablaW mismatch at filter " << filter << ", row " << filterRow << ", column " << filterCol;
			  Assert::AreEqual(expectedWeightError, nablaW.Get(filter, channel, filterRow, filterCol), 1e-5, msg.str().c_str());
			}
		  }
		}

		double expectedBiasError = 0.0;
		for (int deltaRow = 0; deltaRow < 4; ++deltaRow)
		{
		  for (int deltaCol = 0; deltaCol < 4; ++deltaCol)
			expectedBiasError += delta[filter * 16 + (deltaRow * 4) + deltaCol];
		}
		std::wostringstream msg;
		msg << "NablaB mismatch at filter " << filter;
		Assert::AreEqual(expectedBiasError, nablaB.Get(filter), 1e-5, msg.str().c_str());
	  }
	}

	TEST_METHOD(FourChannelTriple3x3FilterStride2ZeroPad2ConvolutionalLayerUpdateWeightAndBiasErrors)
	{
	  double delta[3 * 4 * 4] =
	  {
		0.12607, -0.72235, -0.47865, 0.03274,
		0.69829, 0.22116, 0.55568, -0.98420,
		-0.86539, -0.55813, 0.39248, 0.03969,
		0.94210, 0.70328, 0.57449, 0.45554,
		-0.93831, 0.99044, 0.03875, -0.47329,
		-0.77945, 0.56086, -0.60250, -0.88574,
		-0.04890, 0.10612, 0.51434, -0.85138,
		0.50008, 0.76882, 0.09435, -0.99368,
		-0.96563, 0.80097, -0.20009, 0.16932,
		-0.11519, -0.83567, 0.40479, 0.03916,
		0.85537, 0.50877, -0.33722, 0.20106,
		-0.59736, 0.55277, -0.66022, 0.09966
	  };
	  Tensor deltaTensor(std::initializer_list<double>(delta, delta + (3 * 16)), 3, 4, 4);
	  double prevActivations[4 * 5 * 5] =
	  {
		-0.56507, 0.36559, -0.35899, -0.27600, 0.05529,
		0.62505, -0.44365, -0.17203, -0.77168, 0.19406,
		-0.04839, -0.46031, -0.54320, -0.56026, -0.83502,
		0.34759, -0.45779, 0.05624, 0.17175, 0.95325,
		-0.73179, 0.40213, 0.15014, -0.40278, 0.47441,
		0.16861, 0.14238, -0.36656, 0.60456, -0.05017,
		-0.90399, 0.52897, -0.07747, 0.37172, -0.21245,
		0.69106, -0.09681, 0.23668, -0.31201, 0.13491,
		0.59159, -0.57499, 0.27150, 0.10668, -0.59564,
		-0.55303, -0.51030, 0.57432, -0.65156, -0.34578,
		0.32573, -0.05459, -0.79745, -0.26688, 0.73741,
		0.52317, -0.15901, -0.41513, 0.63699, -0.39244,
		-0.71243, 0.37299, -0.56435, -0.55147, 0.56429,
		-0.37949, 0.46795, -0.98657, 0.85911, 0.71236,
		0.06849, -0.29096, 0.41458, 0.91817, -0.10302,
		0.61194, -0.97058, -0.50139, 0.42106, -0.35197,
		-0.43951, 0.27664, 0.82793, -0.36162, -0.25591,
		-0.94994, 0.50386, -0.57323, -0.06200, 0.20186,
		0.62478, -0.96724, -0.39057, -0.10762, -0.53184,
		-0.53545, 0.66465, 0.84454, 0.64909, 0.16745
	  };
	  Tensor prevActivationsTensor(std::initializer_list<double>(prevActivations, prevActivations + 4 * 5 * 5), 4, 5, 5);

	  ConvolutionalLayer layer(4, 5, 5, 3, 3, 2, 2, nullptr);
	  layer.InitializeWeights();
	  Tensor nablaW(3, 4, 3, 3);
	  Tensor nablaB(3);
	  layer.UpdateWeightAndBiasErrors(deltaTensor, prevActivationsTensor, nablaW, nablaB, nullptr);

	  for (int filter = 0; filter < 3; ++filter)
	  {
		for (int channel = 0; channel < 4; ++channel)
		{
		  for (int filterRow = 0; filterRow < 3; ++filterRow)
		  {
			for (int filterCol = 0; filterCol < 3; ++filterCol)
			{
			  double expectedWeightError = 0.0;
			  double* del = delta + (filter * 16);
			  for (int filterRowOffset = -2; filterRowOffset <= 4; filterRowOffset += 2)
			  {
				int prevRow = filterRowOffset + filterRow;
				for (int filterColOffset = -2; filterColOffset <= 4; filterColOffset += 2)
				{
				  int prevCol = filterColOffset + filterCol;
				  if (prevRow >= 0 && prevRow < 5 && prevCol >= 0 && prevCol < 5)
				  {
					double activation = prevActivations[(channel * 5 * 5) + prevRow * 5 + prevCol];
					expectedWeightError += (*del * activation);
				  }
				  ++del;
				}
			  }

			  std::wostringstream msg;
			  msg << "NablaW mismatch at filter " << filter << ", row " << filterRow << ", column " << filterCol;
			  Assert::AreEqual(expectedWeightError, nablaW.Get(filter, channel, filterRow, filterCol), 1e-5, msg.str().c_str());
			}
		  }
		}

		double expectedBiasError = 0.0;
		for (int deltaRow = 0; deltaRow < 4; ++deltaRow)
		{
		  for (int deltaCol = 0; deltaCol < 4; ++deltaCol)
			expectedBiasError += delta[filter * 16 + (deltaRow * 4) + deltaCol];
		}
		std::wostringstream msg;
		msg << "NablaB mismatch at filter " << filter;
		Assert::AreEqual(expectedBiasError, nablaB.Get(filter), 1e-5, msg.str().c_str());
	  }
	}

	TEST_METHOD(SingleChannelTriple4x4FilterStride2ConvolutionalLayerUpdateWeightAndBiasErrors)
	{
	  double delta[3 * 6 * 6] =
	  {
		-0.96678, -0.78373, -0.55794, 0.61000, 0.15480, 0.54525,
		0.51857, 0.98925, 0.73807, -0.21690, 0.22571, 0.74447,
		-0.84138, 0.46042, 0.72620, -0.06277, 0.83815, 0.71590,
		0.25999, -0.21358, 0.07483, 0.68542, 0.35257, 0.65511,
		-0.31429, 0.49485, 0.37355, -0.72522, -0.05983, -0.85455,
		0.73681, 0.01262, 0.47648, -0.81299, 0.82210, -0.52620,
		0.46272, -0.43281, -0.23503, 0.20874, -0.34779, -0.04097,
		-0.30844, 0.98802, 0.67215, -0.88021, 0.53604, -0.50555,
		-0.13694, 0.79932, -0.34098, -0.17982, 0.27516, 0.85690,
		0.29730, 0.73544, -0.81252, 0.21409, -0.15505, 0.03118,
		-0.91480, -0.90857, -0.18155, 0.64351, -0.24117, -0.07476,
		-0.24550, -0.50893, -0.42020, 0.32371, -0.92890, -0.67828,
		0.70364, -0.67919, -0.03534, 0.22645, -0.26145, -0.07216,
		0.29211, 0.69104, -0.07076, -0.89246, -0.60748, 0.98246,
		0.38547, 0.09226, -0.81086, -0.37228, -0.77526, 0.55240,
		0.20032, -0.83777, 0.74169, 0.58163, -0.74490, 0.11393,
		0.10815, 0.23209, 0.41872, -0.86880, -0.98406, 0.29992,
		-0.93250, -0.14714, -0.05385, 0.28263, -0.77603, 0.16565
	  };
	  Tensor deltaTensor(std::initializer_list<double>(delta, delta + (3 * 6 * 6)), 3, 6, 6);
	  double prevActivations[14 * 14] =
	  {
		-0.45717, 0.56647, -0.19666, -0.52141, 0.23367, -0.39150, -0.24228, -0.73616, -0.12727, 0.40256, 0.93656, -0.20966, 0.89261, -0.31925,
		0.89022, -0.21275, 0.67976, 0.03999, 0.08298, 0.71521, -0.27732, 0.84380, 0.32487, 0.48214, -0.83854, -0.79588, 0.18181, -0.67166,
		0.17738, 0.23330, -0.39747, -0.80519, -0.17441, -0.81872, -0.99036, -0.19167, -0.36131, -0.95991, -0.89207, 0.09318, -0.46667, -0.50021,
		0.11590, -0.68496, -0.29065, -0.79376, -0.83544, 0.81274, 0.74917, 0.15883, -0.34096, -0.85089, -0.14817, -0.58196, 0.72400, 0.84202,
		-0.51843, -0.00068, -0.99172, -0.06092, -0.92530, -0.39218, 0.10409, -0.94662, 0.92546, 0.10457, -0.33008, 0.37453, -0.01787, -0.24379,
		-0.43293, -0.16701, 0.56092, 0.65028, 0.40317, 0.95208, 0.03645, -0.45435, 0.02843, 0.33101, -0.23886, -0.90083, 0.99633, 0.18024,
		-0.26672, -0.32798, -0.98618, -0.97406, 0.20561, 0.67423, -0.83725, -0.88813, 0.97991, -0.92622, -0.71476, -0.70485, 0.34833, -0.23390,
		-0.46723, -0.43604, 0.46067, -0.79202, -0.24506, 0.30052, 0.78051, 0.80025, 0.13179, -0.62009, 0.74899, -0.23025, 0.41448, 0.86542,
		-0.77440, -0.65291, -0.73744, 0.89786, -0.55413, 0.71369, 0.06749, -0.68401, 0.68886, 0.02452, -0.47295, 0.67186, -0.94570, 0.32079,
		-0.53325, -0.78304, 0.49515, 0.36337, -0.96134, -0.13324, -0.45708, 0.22442, 0.62262, -0.03454, -0.04635, -0.37782, -0.58480, -0.13091,
		-0.25255, -0.33273, -0.92969, -0.13519, 0.59695, -0.90043, -0.25303, -0.67953, -0.85265, -0.33727, -0.18508, 0.56905, -0.52552, -0.60576,
		0.37576, 0.77516, 0.46387, -0.30350, 0.58669, -0.64285, 0.39470, -0.76126, 0.19952, -0.22954, 0.60378, -0.08105, 0.80178, 0.91239,
		-0.92570, 0.67442, -0.56418, 0.84283, -0.93858, -0.79864, 0.42585, -0.06177, 0.29975, 0.96870, -0.94522, -0.09136, 0.09157, 0.56674,
		-0.32820, -0.31908, 0.87992, 0.38546, 0.93386, 0.13740, -0.10707, -0.08691, -0.85815, -0.48607, 0.58509, 0.47747, 0.31692, 0.04579
	  };
	  Tensor prevActivationsTensor(std::initializer_list<double>(prevActivations, prevActivations + (14 * 14)), 1, 14, 14);

	  ConvolutionalLayer layer(1, 14, 14, 3, 4, 2, 0, nullptr);
	  layer.InitializeWeights();
	  Tensor nablaW(3, 1, 4, 4);
	  Tensor nablaB(3);
	  layer.UpdateWeightAndBiasErrors(deltaTensor, prevActivationsTensor, nablaW, nablaB, nullptr);

	  for (int filter = 0; filter < 3; ++filter)
	  {
		for (int filterRow = 0; filterRow < 4; ++filterRow)
		{
		  for (int filterCol = 0; filterCol < 4; ++filterCol)
		  {
			double expectedWeightError = 0.0;
			for (int deltaRow = 0; deltaRow < 6; ++deltaRow)
			{
			  for (int deltaCol = 0; deltaCol < 6; ++deltaCol)
			  {
				double del = delta[filter * 36 + deltaRow * 6 + deltaCol];
				uint32_t inputRow = deltaRow * 2 + filterRow;
				uint32_t inputCol = deltaCol * 2 + filterCol;
				double activation = prevActivations[inputRow * 14 + inputCol];
				expectedWeightError += (del * activation);
			  }
			}
			std::wostringstream msg;
			msg << "NablaW mismatch at filter " << filter << ", row " << filterRow << ", column " << filterCol;
			Assert::AreEqual(expectedWeightError, nablaW.Get(filter, 0, filterRow, filterCol), 1e-5, msg.str().c_str());
		  }
		}

		double expectedBiasError = 0.0;
		for (int deltaRow = 0; deltaRow < 6; ++deltaRow)
		{
		  for (int deltaCol = 0; deltaCol < 6; ++deltaCol)
			expectedBiasError += delta[filter * 36 + deltaRow * 6 + deltaCol];
		}
		std::wostringstream msg;
		msg << "NablaB mismatch at filter " << filter;
		Assert::AreEqual(expectedBiasError, nablaB.Get(filter), 1e-5, msg.str().c_str());
	  }
	}

	TEST_METHOD(SingleChannelTriple4x4FilterStride2ZeroPad2ConvolutionalLayerUpdateWeightAndBiasErrors)
	{
	  double delta[3 * 6 * 6] =
	  {
		-0.96678, -0.78373, -0.55794, 0.61000, 0.15480, 0.54525,
		0.51857, 0.98925, 0.73807, -0.21690, 0.22571, 0.74447,
		-0.84138, 0.46042, 0.72620, -0.06277, 0.83815, 0.71590,
		0.25999, -0.21358, 0.07483, 0.68542, 0.35257, 0.65511,
		-0.31429, 0.49485, 0.37355, -0.72522, -0.05983, -0.85455,
		0.73681, 0.01262, 0.47648, -0.81299, 0.82210, -0.52620,
		0.46272, -0.43281, -0.23503, 0.20874, -0.34779, -0.04097,
		-0.30844, 0.98802, 0.67215, -0.88021, 0.53604, -0.50555,
		-0.13694, 0.79932, -0.34098, -0.17982, 0.27516, 0.85690,
		0.29730, 0.73544, -0.81252, 0.21409, -0.15505, 0.03118,
		-0.91480, -0.90857, -0.18155, 0.64351, -0.24117, -0.07476,
		-0.24550, -0.50893, -0.42020, 0.32371, -0.92890, -0.67828,
		0.70364, -0.67919, -0.03534, 0.22645, -0.26145, -0.07216,
		0.29211, 0.69104, -0.07076, -0.89246, -0.60748, 0.98246,
		0.38547, 0.09226, -0.81086, -0.37228, -0.77526, 0.55240,
		0.20032, -0.83777, 0.74169, 0.58163, -0.74490, 0.11393,
		0.10815, 0.23209, 0.41872, -0.86880, -0.98406, 0.29992,
		-0.93250, -0.14714, -0.05385, 0.28263, -0.77603, 0.16565
	  };
	  Tensor deltaTensor(std::initializer_list<double>(delta, delta + (3 * 6 * 6)), 3, 6, 6);
	  double prevActivations[10 * 10] =
	  {
		-0.45717, 0.56647, -0.19666, -0.52141, 0.23367, -0.39150, -0.24228, -0.73616, -0.12727, 0.40256,
		0.89022, -0.21275, 0.67976, 0.03999, 0.08298, 0.71521, -0.27732, 0.84380, 0.32487, 0.48214,
		0.17738, 0.23330, -0.39747, -0.80519, -0.17441, -0.81872, -0.99036, -0.19167, -0.36131, -0.95991,
		0.11590, -0.68496, -0.29065, -0.79376, -0.83544, 0.81274, 0.74917, 0.15883, -0.34096, -0.85089,
		-0.51843, -0.00068, -0.99172, -0.06092, -0.92530, -0.39218, 0.10409, -0.94662, 0.92546, 0.10457,
		-0.53325, -0.78304, 0.49515, 0.36337, -0.96134, -0.13324, -0.45708, 0.22442, 0.62262, -0.13091,
		-0.25255, -0.33273, -0.92969, -0.13519, 0.59695, -0.90043, -0.25303, -0.67953, -0.85265, -0.33727,
		0.37576, 0.77516, 0.46387, -0.30350, 0.58669, -0.64285, 0.39470, -0.76126, 0.19952, -0.22954,
		0.67442, -0.56418, 0.84283, -0.93858, -0.79864, 0.42585, -0.06177, 0.29975, 0.96870, -0.94522,
		0.93386, 0.13740, -0.10707, -0.08691, -0.85815, -0.48607, 0.58509, 0.47747, 0.31692, 0.04579
	  };
	  Tensor prevActivationsTensor(std::initializer_list<double>(prevActivations, prevActivations + (10 * 10)), 1, 10, 10);

	  ConvolutionalLayer layer(1, 10, 10, 3, 4, 2, 2, nullptr);
	  layer.InitializeWeights();
	  Tensor nablaW(3, 1, 4, 4);
	  Tensor nablaB(3);
	  layer.UpdateWeightAndBiasErrors(deltaTensor, prevActivationsTensor, nablaW, nablaB, nullptr);

	  for (int filter = 0; filter < 3; ++filter)
	  {
		for (int filterRow = 0; filterRow < 4; ++filterRow)
		{
		  for (int filterCol = 0; filterCol < 4; ++filterCol)
		  {
			double expectedWeightError = 0.0;
			const double* del = delta + (filter * 36);
			for (int filterRowOffset = -2; filterRowOffset <= 8; filterRowOffset += 2)
			{
			  int prevRow = filterRowOffset + filterRow;
			  for (int filterColOffset = -2; filterColOffset <= 8; filterColOffset += 2)
			  {
				int prevCol = filterColOffset + filterCol;
				if (prevRow >= 0 && prevRow < 10 && prevCol >= 0 && prevCol < 10)
				{
				  double activation = prevActivations[prevRow * 10 + prevCol];
				  expectedWeightError += (*del * activation);
				}
				++del;
			  }
			}

			std::wostringstream msg;
			msg << "NablaW mismatch at filter " << filter << ", row " << filterRow << ", column " << filterCol;
			Assert::AreEqual(expectedWeightError, nablaW.Get(filter, 0, filterRow, filterCol), 1e-5, msg.str().c_str());
		  }
		}

		double expectedBiasError = 0.0;
		for (int deltaRow = 0; deltaRow < 6; ++deltaRow)
		{
		  for (int deltaCol = 0; deltaCol < 6; ++deltaCol)
			expectedBiasError += delta[filter * 36 + deltaRow * 6 + deltaCol];
		}
		std::wostringstream msg;
		msg << "NablaB mismatch at filter " << filter;
		Assert::AreEqual(expectedBiasError, nablaB.Get(filter), 1e-5, msg.str().c_str());
	  }
	}
  };
}
