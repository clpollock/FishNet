#include "stdafx.h"
#include "ConvolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(TensorPtr&& weights, TensorPtr&& biases,
  uint32_t inputRows, uint32_t inputColumns, uint32_t stride, uint32_t zeroPadding, std::unique_ptr<::ActivationFunction>&& activationFunction)
  : WeightedLayer(std::move(weights), std::move(biases), std::move(activationFunction), weights->Hyperplanes(),
	  (inputRows + (zeroPadding * 2) - weights->Rows()) / stride + 1, (inputColumns + (zeroPadding * 2) - weights->Rows()) / stride + 1),
	_inputChannelCount(_weights->Planes()),
	_inputRows(inputRows),
	_inputColumns(inputColumns),
	_filterCount(_weights->Hyperplanes()),
	_filterSize(_weights->Rows()),
	_stride(stride),
	_zeroPadding(zeroPadding)
{
  if (_zeroPadding >= _filterSize)
	throw std::runtime_error("Zero padding must be less than the size of the filter.");
  if (_filterSize != _weights->Columns())
	throw std::runtime_error("Filter width and height must be the same.");
  if (_filterCount != _biases->Size())
	throw std::runtime_error("There must be 1 bias for each filter.");
  CalculateFilterInfo();
}

ConvolutionalLayer::ConvolutionalLayer(uint32_t inputChannelCount, uint32_t inputRows, uint32_t inputColumns, uint32_t filterCount,
  uint32_t filterSize, uint32_t stride, uint32_t zeroPadding, std::unique_ptr<::ActivationFunction>&& activationFunction)
  : WeightedLayer(std::move(activationFunction), filterCount, (inputRows + (zeroPadding * 2) - filterSize) / stride + 1,
	  (inputColumns + (zeroPadding * 2) - filterSize) / stride + 1),
	_inputChannelCount(inputChannelCount),
	_inputRows(inputRows),
	_inputColumns(inputColumns),
	_filterCount(filterCount),
	_filterSize(filterSize),
	_stride(stride),
	_zeroPadding(zeroPadding)
{
}

void ConvolutionalLayer::InitializeWeights()
{
  if (_weights == nullptr)
  {
	_weights = std::make_unique<Tensor>(_filterCount, _inputChannelCount, _filterSize, _filterSize);
	_biases = std::make_unique<Tensor>(_filterCount);
	// Randomize weights and biases
	Randomizer randomizer(2.0 / sqrt(double(_filterSize * _filterSize * _inputChannelCount)));
	randomizer.Fill(*_weights);
	CalculateFilterInfo();
  }
}

void ConvolutionalLayer::Description(std::ostream& os) const
{
  os << "Convolutional, " << _filterCount << " filters, input dimensions: " << _inputChannelCount << 'x' << _inputColumns << 'x' << _inputRows
	<< ", filter size: " << _filterSize << 'x' << _filterSize << ", stride: " << _stride;
  if (_zeroPadding != 0)
	os << ", zero padding: "	<< _zeroPadding;
  os << ", activation: "	<< (_activationFunction ? _activationFunction->Description() : "None");
}

void ConvolutionalLayer::FeedForward(const Tensor& inputs, Tensor& outputs, const DropoutMask*) const
{
#ifdef _DEBUG
  if (inputs.Planes() != _inputChannelCount)
	throw std::runtime_error("ConvolutionalLayer::FeedForward - input tensor has the wrong number of channels.");
  if (inputs.Rows() != _inputRows)
	throw std::runtime_error("ConvolutionalLayer::FeedForward - input tensor has the wrong number of rows.");
  if (inputs.Columns() != _inputColumns)
	throw std::runtime_error("ConvolutionalLayer::FeedForward - input tensor has the wrong number of columns.");
  if (outputs.Planes() != _filterCount)
	throw std::runtime_error("ConvolutionalLayer::FeedForward - output tensor has the wrong number of channels.");
  if (outputs.Rows() != (_inputRows + (2 * _zeroPadding) - _filterSize) / _stride + 1)
	throw std::runtime_error("ConvolutionalLayer::FeedForward - output tensor has the wrong number of rows.");
  if (outputs.Columns() != (_inputColumns + (2 * _zeroPadding) - _filterSize) / _stride + 1)
	throw std::runtime_error("ConvolutionalLayer::FeedForward - output tensor has the wrong number of columns.");
#endif
  double* output = outputs.Elements();
  if (_zeroPadding > 0)
  {
	int32_t endRow = _inputRows + _zeroPadding - _filterSize + 1;
	int32_t endCol = _inputColumns + _zeroPadding - _filterSize + 1;
	for (uint32_t filter = 0; filter < _filterCount; ++filter)
	{
	  double filterBias = _biases->Get(filter);
	  for (int32_t inputRow = -_zeroPadding; inputRow < endRow; inputRow += _stride)
	  {
		int32_t filterStartRow = 0;
		int32_t filterEndRow = _filterSize;
		if (inputRow < 0)
		{
		  filterStartRow = -inputRow;
		}
		else if (inputRow + _filterSize > _inputRows)
		{
		  filterEndRow = _inputRows - inputRow;
		}
		size_t filterHeightTimesInputWidth = (filterEndRow - filterStartRow) * _inputColumns;
		for (int32_t inputCol = -_zeroPadding; inputCol < endCol; inputCol += _stride)
		{
		  int32_t filterStartCol = 0;
		  int32_t filterEndCol = _filterSize;
		  if (inputCol < 0)
		  {
			filterStartCol = -inputCol;
		  }
		  else if (inputCol + _filterSize > _inputColumns)
		  {
			filterEndCol = _inputColumns - inputCol;
		  }
		  size_t filterWidth = filterEndCol - filterStartCol;
		  size_t filterRowOffset = _filterSize - filterWidth;
		  size_t rowOffset = _inputColumns - filterWidth;
		  double activation = filterBias;
		  for (uint32_t inputChannel = 0; inputChannel < _inputChannelCount; ++inputChannel)
		  {
			const double* weight = _weights->ElementAddress(filter, inputChannel, filterStartRow, filterStartCol);
			const double* in = inputs.ElementAddress(inputChannel, std::max(inputRow, 0), std::max(inputCol, 0));
			const double* inEnd = in + filterHeightTimesInputWidth;
			do
			{
			  const double* weightRowEnd = weight + filterWidth;
			  do
			  {
				activation += (*in * *weight);
				++weight;
				++in;
			  } while (weight != weightRowEnd);
			  weight += filterRowOffset;
			  in += rowOffset;
			} while (in < inEnd);
		  }
		  *output = activation;
		  ++output;
		}
	  }
	}
  }
  else
  {
	uint32_t rowOffset = _inputColumns - _filterSize;
	uint32_t endRow = _inputRows - _filterSize + 1;
	uint32_t endCol = rowOffset + 1;
	uint32_t filterSizeTimesInputWidth = _filterSize * _inputColumns;

	const double* filterWeights = _weights->Elements();
	size_t filterWeightSize = _weights->Planes() * _weights->Rows() * _weights->Columns();
	for (uint32_t filter = 0; filter < _filterCount; ++filter)
	{
	  double filterBias = _biases->Get(filter);
	  for (uint32_t inputRow = 0; inputRow < endRow; inputRow += _stride)
	  {
		for (uint32_t inputCol = 0; inputCol < endCol; inputCol += _stride)
		{
		  double activation = filterBias;
		  const double* weight = filterWeights;
		  for (uint32_t inputChannel = 0; inputChannel < _inputChannelCount; ++inputChannel)
		  {
			const double* in = inputs.ElementAddress(inputChannel, inputRow, inputCol);
			const double* inEnd = in + filterSizeTimesInputWidth;
			do
			{
			  const double* weightRowEnd = weight + _filterSize;
			  do
			  {
				activation += (*in * *weight);
				++weight;
				++in;
			  } while (weight != weightRowEnd);
			  in += rowOffset;
			} while (in < inEnd);
		  }
		  *output = activation;
		  ++output;
		}
	  }
	  filterWeights += filterWeightSize;
	}
  }
}

void ConvolutionalLayer::Save(std::ofstream& os) const
{
  os.put((char)Types::Convolutional);
  if(_activationFunction)
	_activationFunction->Save(os);
  else
	os.put((char)ActivationFunction::Types::None);
  os.write((const char*)&_stride, sizeof(uint32_t));
  os.write((const char*)&_zeroPadding, sizeof(uint32_t));
  _weights->Save(os);
  _biases->Save(os);
}

void ConvolutionalLayer::SaveArchitecture(std::ostream& os) const
{
  os << "Convolutional,," << _filterCount << ',' << _filterSize << ',' << _stride << ',' << _zeroPadding << ",,";
  if (_activationFunction)
  {
	os << _activationFunction->Name();
	const auto* lru = dynamic_cast<const LeakyReLU*>(_activationFunction.get());
	if (lru)
	  os << ',' << lru->Leakiness();
	os << std::endl;
  }
}

void ConvolutionalLayer::BackpropagateError(const Tensor& errorInThisLayer, Tensor& errorInPreviousLayer, const DropoutMask*) const
{
#ifdef _DEBUG
  if (errorInPreviousLayer.Planes() != _inputChannelCount)
	throw std::runtime_error("ConvolutionalLayer::BackpropagateError - error in previous layer tensor has the wrong number of channels.");
  if (errorInPreviousLayer.Rows() != _inputRows)
	throw std::runtime_error("ConvolutionalLayer::BackpropagateError - error in previous layer tensor has the wrong number of rows.");
  if (errorInPreviousLayer.Columns() != _inputColumns)
	throw std::runtime_error("ConvolutionalLayer::BackpropagateError - error in previous layer tensor has the wrong number of columns.");
  if (errorInThisLayer.Planes() != _filterCount)
	throw std::runtime_error("ConvolutionalLayer::BackpropagateError - error in this layer tensor has the wrong number of channels.");
  if (errorInThisLayer.Rows() != (_inputRows + (_zeroPadding * 2) - _filterSize) / _stride + 1)
	throw std::runtime_error("ConvolutionalLayer::BackpropagateError - error in this layer tensor has the wrong number of rows.");
  if (errorInThisLayer.Columns() != (_inputColumns + (_zeroPadding * 2) - _filterSize) / _stride + 1)
	throw std::runtime_error("ConvolutionalLayer::BackpropagateError - error in this layer tensor has the wrong number of columns.");
#endif
  uint32_t outputRows = errorInThisLayer.Rows();
  uint32_t outputCols = errorInThisLayer.Columns();
  errorInPreviousLayer.SetAllToZero();
  if (_zeroPadding > 0)
  {
	for (uint32_t filter = 0; filter < _filterCount; ++filter)
	{
	  for (uint32_t inputChannel = 0; inputChannel < _inputChannelCount; ++inputChannel)
	  {
		const double* outputError = errorInThisLayer.ElementAddress(filter, 0, 0);
		int32_t inputRow = -_zeroPadding;
		for (uint32_t outputRow = 0; outputRow < outputRows; ++outputRow)
		{
		  int32_t filterStartRow = 0;
		  int32_t filterEndRow = _filterSize;
		  if (inputRow < 0)
		  {
			filterStartRow = -inputRow;
		  }
		  else if (inputRow + _filterSize > _inputRows)
		  {
			filterEndRow = _inputRows - inputRow;
		  }

		  int32_t inputCol = -_zeroPadding;
		  for (uint32_t outputCol = 0; outputCol < outputCols; ++outputCol)
		  {
			int32_t filterStartCol = 0;
			int32_t filterEndCol = _filterSize;
			if (inputCol < 0)
			{
			  filterStartCol = -inputCol;
			}
			else if (inputCol + _filterSize > _inputColumns)
			{
			  filterEndCol = _inputColumns - inputCol;
			}
			size_t filterWidth = filterEndCol - filterStartCol;
			size_t filterRowOffset = _filterSize - filterWidth;
			size_t inputRowOffset = _inputColumns - filterWidth;

			const double* weight = _weights->ElementAddress(filter, inputChannel, filterStartRow, filterStartCol);
			// 
			double* prevError = errorInPreviousLayer.ElementAddress(inputChannel, std::max(0, inputRow), std::max(0, inputCol));
			for (int32_t filterRow = filterStartRow; filterRow < filterEndRow; ++filterRow)
			{
			  for (int32_t filterCol = filterStartCol; filterCol < filterEndCol; ++filterCol)
			  {
				*prevError += (*outputError * *weight);
				++weight;
				++prevError;
			  }
			  prevError += inputRowOffset;
			  weight += filterRowOffset;
			}
			++outputError;
			inputCol += _stride;
		  }
		  inputRow += _stride;
		}
	  }
	}
  }
  else
  {
	double* inputChannelWeights = _weights->Elements();
	size_t inputChannelWeightSize = _weights->Rows() * _weights->Columns();
    uint32_t inputRowOffset = _inputColumns - _filterSize;
	for (uint32_t filter = 0; filter < _filterCount; ++filter)
	{
	  for (uint32_t inputChannel = 0; inputChannel < _inputChannelCount; ++inputChannel)
	  {
		const double* outputError = errorInThisLayer.ElementAddress(filter, 0, 0);
		uint32_t inputRow = 0;
		for (uint32_t outputRow = 0; outputRow < outputRows; ++outputRow)
		{
  		  uint32_t inputCol = 0;
		  for (uint32_t outputCol = 0; outputCol < outputCols; ++outputCol)
		  {
			double* weight = inputChannelWeights;
			double* prevError = errorInPreviousLayer.ElementAddress(inputChannel, inputRow, inputCol);
			for (int32_t filterRow = 0; filterRow < _filterSize; ++filterRow)
			{
			  for (int32_t filterCol = 0; filterCol < _filterSize; ++filterCol)
			  {
				*prevError += (*outputError * *weight);
				++weight;
				++prevError;
			  }
			  prevError += inputRowOffset;
			}
			++outputError;
			inputCol += _stride;
		  }
		  inputRow += _stride;
		}
		inputChannelWeights += inputChannelWeightSize;
	  }
	}
  }
}

void ConvolutionalLayer::UpdateWeightAndBiasErrors(const Tensor& delta, const Tensor& previousLayerActivations,
  Tensor& nablaW, Tensor& nablaB, const DropoutMask*)
{
#ifdef _DEBUG
  if (previousLayerActivations.Planes() != _inputChannelCount)
	throw std::runtime_error("ConvolutionalLayer::BackpropagateError - previous layer activations tensor has the wrong number of channels.");
  if (previousLayerActivations.Rows() != _inputRows)
	throw std::runtime_error("ConvolutionalLayer::BackpropagateError - previous layer activations tensor has the wrong number of rows.");
  if (previousLayerActivations.Columns() != _inputColumns)
	throw std::runtime_error("ConvolutionalLayer::BackpropagateError - previous layer activations tensor has the wrong number of columns.");
  if (!nablaW.DimensionsMatch(*_weights))
	throw std::runtime_error("ConvolutionalLayer::UpdateWeightAndBiasErrors - Dimensions of nablaW do not match the weight dimensions.");
  if (!nablaB.DimensionsMatch(*_biases))
	throw std::runtime_error("FullyConnectedLayer::UpdateWeightAndBiasErrors - Dimensions of nablaB do not match the bias dimensions.");
#endif
  uint32_t deltaPlaneSize = delta.Rows() * delta.Columns();
  uint32_t inputWidthTimesStride = _inputColumns * _stride;
  double* thisNablaW = nablaW.Elements();
  double* thisNablaB = nablaB.Elements();
  for (uint32_t filter = 0; filter < _filterCount; ++filter)
  {
	if (_zeroPadding > 0)
	{
	  for (uint32_t inputChannel = 0; inputChannel < _inputChannelCount; ++inputChannel)
	  {
		const FilterInfo* filterRowInfo = _filterRowInfo.get();
		for (int32_t filterRow = 0; filterRow < _filterSize; ++filterRow)
		{
		  int32_t activationRowOffset = filterRowInfo->outputSpan * _stride * _inputColumns;
		  const FilterInfo* filterColInfo = _filterColumnInfo.get();
		  for (int32_t filterCol = 0; filterCol < _filterSize; ++filterCol)
		  {
			int32_t inputColSpan = filterColInfo->outputSpan * _stride;
			const double* activation = previousLayerActivations.ElementAddress(inputChannel,
			  filterRowInfo->inputStartOffset, filterColInfo->inputStartOffset);
			const double* rowActivationEnd = activation + inputColSpan;
			size_t activationOffset = inputWidthTimesStride - inputColSpan;
			const double* activationEnd = activation + activationRowOffset;
			const double* del = delta.ElementAddress(filter, filterRowInfo->outputStartOffset, filterColInfo->outputStartOffset);
			int32_t delOffset = delta.Columns() - filterColInfo->outputSpan;
			double thisError = 0.0;
			// Iterate over input rows.
			do
			{
			  // Iterate over input columns.
			  do
			  {
				thisError += (*del * *activation);
				++del;
				activation += _stride;
			  } while (activation < rowActivationEnd);
			  activation += activationOffset;
			  rowActivationEnd += inputWidthTimesStride;
			  del += delOffset;
			} while (activation < activationEnd);
			*thisNablaW += thisError;
			++thisNablaW;
			++filterColInfo;
		  }
		  ++filterRowInfo;
		}
	  }
	}
	else
	{
	  for (uint32_t inputChannel = 0; inputChannel < _inputChannelCount; ++inputChannel)
	  {
		for (int32_t filterRow = 0; filterRow < _filterSize; ++filterRow)
		{
		  for (int32_t filterCol = 0; filterCol < _filterSize; ++filterCol)
		  {
			const double* rowFirstActivation = previousLayerActivations.ElementAddress(inputChannel, filterRow, filterCol);
			const double* del = delta.ElementAddress(filter, 0, 0);
			const double* delEnd = del + deltaPlaneSize;
			double thisError = 0.0;
			// Iterate over input rows.
			do
			{
			  const double* activation = rowFirstActivation;
			  const double* delRowEnd = del + delta.Columns();
			  // Iterate over input columns.
			  do
			  {
				thisError += (*del * *activation);
				++del;
				activation += _stride;
			  } while (del < delRowEnd);
			  rowFirstActivation += inputWidthTimesStride;
			} while (del < delEnd);
			*thisNablaW += thisError;
			++thisNablaW;
		  }
		}
	  }
	}

	double biasUpdate = 0.0;
	const double* del = delta.ElementAddress(filter, 0, 0);
	const double* delEnd = del + deltaPlaneSize;
	do
	{
	  biasUpdate += *del;
	  ++del;
	} while (del < delEnd);

	*thisNablaB += biasUpdate;
	++thisNablaB;
  }
}

void ConvolutionalLayer::CalculateFilterInfo()
{
  _filterRowInfo = std::make_unique<FilterInfo[]>(_filterSize);
  _filterColumnInfo = std::make_unique<FilterInfo[]>(_filterSize);
  CalculateFilterInfo(_filterRowInfo.get(), _inputRows);
  CalculateFilterInfo(_filterColumnInfo.get(), _inputColumns);
}

void ConvolutionalLayer::CalculateFilterInfo(ConvolutionalLayer::FilterInfo* filterInfo, int32_t inputDimensionLength)
{
  for (int32_t i = 0; i < _zeroPadding; ++i)
  {
	int32_t offset = _zeroPadding - i;
	filterInfo[i].inputStartOffset = offset % _stride;
	filterInfo[i].outputStartOffset = offset / _stride;
	if (offset % _stride)
	  ++filterInfo[i].outputStartOffset;
  }

  for (int32_t i = _zeroPadding; i < _filterSize; ++i)
  {
	filterInfo[i].inputStartOffset = i - _zeroPadding;
	filterInfo[i].outputStartOffset = 0;
  }

  // This code is ugly and inefficient, but I have no time to fix it now.
  // It is only called when networks are initialized, so the actual effect
  // on performance is trivial.
  int32_t outputOffset = 0;
  for (int32_t offset = -_zeroPadding; offset + _filterSize <= inputDimensionLength + _zeroPadding; offset += _stride)
  {
	for (int32_t i = 0; i < std::min(_filterSize, inputDimensionLength - offset); ++i)
	{
	  filterInfo[i].outputSpan = outputOffset - filterInfo[i].outputStartOffset + 1;
	}
	++outputOffset;
  }
}
