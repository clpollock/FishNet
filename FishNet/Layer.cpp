#include "stdafx.h"
#include "ConvolutionalLayer.h"
#include "DropoutMask.h"

void Randomizer::Fill(Tensor& tensor)
{
  double* dest = tensor.Elements();
  const double* end = dest + tensor.Size();
  while (dest != end)
  {
	*dest = _distribution(_generator);
	++dest;
  }
}

std::default_random_engine Randomizer::_generator(static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count()));

std::unique_ptr<Layer> Layer::Load(std::ifstream& is, uint32_t inputChannelCount, uint32_t inputRows, uint32_t inputColumns,
  double prevLayerKeepProbability)
{
  switch(static_cast<Types>(is.get()))
  {
	case Types::FullyConnected:
	{
	  auto activationFunction = ActivationFunction::Load(is);
	  double keepProbability;
	  is.read((char*)&keepProbability, sizeof(double));
	  auto weights = Tensor::Load(is);
	  auto biases = Tensor::Load(is);
	  return std::make_unique<FullyConnectedLayer>(std::move(weights), std::move(biases), std::move(activationFunction),
		keepProbability, prevLayerKeepProbability);
	}
	case Types::Convolutional:
	{
	  auto activationFunction = ActivationFunction::Load(is);
	  uint32_t stride;
	  is.read((char*)&stride, sizeof(uint32_t));
	  uint32_t zeroPadding;
	  is.read((char*)&zeroPadding, sizeof(uint32_t));
	  auto weights = Tensor::Load(is);
	  auto biases = Tensor::Load(is);
	  return std::make_unique<ConvolutionalLayer>(std::move(weights), std::move(biases), inputRows, inputColumns, stride, zeroPadding,
		std::move(activationFunction));
	}
	case Types::MaxPooling:
	  return std::make_unique<MaxPoolingLayer>(inputChannelCount, inputRows, inputColumns);
	default:
	  throw std::runtime_error("Unrecognized layer type.");
  }
}

void WeightedLayer::UpdateWeightsAndBiases(const Tensor& nablaW, const Tensor& nablaB, double scalar)
{
#ifdef _DEBUG
  if (!nablaW.DimensionsMatch(*_weights))
	throw std::runtime_error("WeightedLayer::UpdateWeightsAndBiases - Dimensions of nablaW do not match the weight dimensions.");
  if (!nablaB.DimensionsMatch(*_biases))
	throw std::runtime_error("WeightedLayer::UpdateWeightsAndBiases - Dimensions of nablaB do not match the bias dimensions.");
#endif
  // Update weights.
  const double* end = _weights->Elements() + _weights->Size();
  double* nw = nablaW.Elements();
  for (double* w = _weights->Elements(); w < end; ++w, ++nw)
	*w -= (*nw * scalar);
  // Update biases.
  end = _biases->Elements() + _biases->Size();
  double* nb = nablaB.Elements();
  for (double* b = _biases->Elements(); b < end; ++b, ++nb)
	*b -= (*nb * scalar);
}

void WeightedLayer::DecayWeights(double factor)
{
  const double* end = _weights->Elements() + _weights->Size();
  for (double* w = _weights->Elements(); w < end; ++w)
	*w *= factor;
}

FullyConnectedLayer::FullyConnectedLayer(TensorPtr&& weights, TensorPtr&& biases, std::unique_ptr<::ActivationFunction>&& activationFunction,
  double keepProbability, double prevLayerKeepProbability)
  : WeightedLayer(std::move(weights), std::move(biases), std::move(activationFunction), 1, 1, weights->Rows()),
	_prevLayerKeepProbability(prevLayerKeepProbability),
	_keepProbability(keepProbability), _inputSize(_weights->Columns())
{
  if (_weights->Hyperplanes() != 1 || _weights->Planes() != 1)
	throw std::runtime_error("FullyConnectedLayer requires a 2 dimensional weight tensor.");
  if (prevLayerKeepProbability < 1.0)
	_trainingWeights = std::make_unique<Tensor>(*_weights);
}

FullyConnectedLayer::FullyConnectedLayer(uint32_t inputSize, uint32_t layerSize, std::unique_ptr<::ActivationFunction>&& activationFunction,
  double keepProbability, double prevLayerKeepProbability)
  : WeightedLayer(std::move(activationFunction), 1, 1, layerSize),
	_prevLayerKeepProbability(prevLayerKeepProbability),
	_keepProbability(keepProbability), _inputSize(inputSize)
{
}

void FullyConnectedLayer::InitializeWeights()
{
  if (_weights == nullptr)
  {
	_weights = std::make_unique<Tensor>(_outputColumns, _inputSize);
	_biases = std::make_unique<Tensor>(_outputColumns);
	// Randomize weights and biases
	Randomizer randomizer(1.0 / sqrt((double)_inputSize));
	randomizer.Fill(*_weights);
	randomizer.Fill(*_biases);
	if (_prevLayerKeepProbability < 1.0)
	  _trainingWeights = std::make_unique<Tensor>(*_weights);
  }
}

void FullyConnectedLayer::Description(std::ostream& os) const
{
  os << "Fully Connected, input size: " << _inputSize << ", output size: " << _outputColumns << ", activation: "
	<< (_activationFunction ? _activationFunction->Description() : "None");
  if (_keepProbability < 1.0)
	os << ", dropout with probability " << (1.0 - _keepProbability);
}

void FullyConnectedLayer::FeedForward(const Tensor& inputs, Tensor& outputs, const DropoutMask* dropoutMask) const
{
#ifdef _DEBUG
  if (inputs.Size() != _weights->Columns())
	throw std::runtime_error("FullyConnectedLayer::FeedForward - Input tensor is the wrong size.");
  if (outputs.Size() != _weights->Rows())
	throw std::runtime_error("FullyConnectedLayer::FeedForward - Output tensor is the wrong size.");
#endif
  const double* weight = _weights->Elements();
  const double* bias = _biases->Elements();
  const double* inputEnd = inputs.Elements() + inputs.Size();
  const double* outputEnd = outputs.Elements() + outputs.Size();
  if (dropoutMask)
  {
	const bool* keep = dropoutMask->Begin();
	for (double* output = outputs.Elements(); output != outputEnd; ++output)
	{
	  if (*keep)
	  {
		double activation = *bias;
		for (const double* input = inputs.Elements(); input != inputEnd; ++input)
		{
		  activation += *input * *weight;
		  ++weight;
		}
		*output = activation;
	  }
	  else
	  {
		*output = 0.0;
		weight += inputs.Size();
	  }
	  ++bias;
	  ++keep;
	}
  }
  else
  {
	for (double* output = outputs.Elements(); output != outputEnd; ++output)
	{
	  double activation = *bias;
	  for (const double* input = inputs.Elements(); input != inputEnd; ++input)
	  {
		activation += *input * *weight;
		++weight;
	  }
	  *output = activation;
	  ++bias;
	}
  }
}

void FullyConnectedLayer::Save(std::ofstream& os) const
{
  os.put((char)Types::FullyConnected);
  if(_activationFunction)
	_activationFunction->Save(os);
  else
	os.put((char)ActivationFunction::Types::None);
  os.write((const char*)&_keepProbability, sizeof(double));
  if (_trainingWeights)
	_trainingWeights->Save(os);
  else
	_weights->Save(os);
  _biases->Save(os);
}

void FullyConnectedLayer::SaveArchitecture(std::ostream& os) const
{
  os << "Fully Connected," << _outputColumns << ",,,,,";
  if (_keepProbability < 1.0)
	os << (1.0 - _keepProbability);
  if (_activationFunction)
  {
	os << ',' << _activationFunction->Name();
	const auto* lru = dynamic_cast<const LeakyReLU*>(_activationFunction.get());
	if (lru)
	  os << ',' << lru->Leakiness();
	os << std::endl;
  }
  os << std::endl;
}

void FullyConnectedLayer::BackpropagateError(const Tensor& errorInThisLayer, Tensor& errorInPreviousLayer, const DropoutMask* dropoutMask) const
{
#ifdef _DEBUG
  if (errorInThisLayer.Size() != _weights->Rows())
	throw std::runtime_error("FullyConnectedLayer::BackpropagateError - Size of errorInThisLayer does not match layer size.");
  if (errorInPreviousLayer.Size() != _weights->Columns())
	throw std::runtime_error("FullyConnectedLayer::BackpropagateError - Size of errorInPreviousLayer does not match input size.");
#endif
  const double* prevLayerErrorEnd = errorInPreviousLayer.Elements() + errorInPreviousLayer.Size();
  const double* thisLayerErrorEnd = errorInThisLayer.Elements() + errorInThisLayer.Size();
  if (dropoutMask)
  {
	errorInPreviousLayer.SetAllToZero();
	const double* weight = _weights->Elements();
	const bool* keep = dropoutMask->Begin();
	for (double* thisLayerError = errorInThisLayer.Elements(); thisLayerError != thisLayerErrorEnd; ++thisLayerError)
	{
	  if (*keep)
	  {
		for (double* prevLayerError = errorInPreviousLayer.Elements(); prevLayerError != prevLayerErrorEnd; ++prevLayerError)
		{
		  *prevLayerError += (*weight * *thisLayerError);
		  ++weight;
		}
	  }
	  else
	  {
		weight += _weights->Columns();
	  }
	  ++keep;
	}
  }
  else
  {
	const double* weightColumnStart = _weights->Elements();
	for (double* prevLayerError = errorInPreviousLayer.Elements(); prevLayerError != prevLayerErrorEnd; ++prevLayerError)
	{
	  const double* weight = weightColumnStart;
	  double error = 0.0;
	  for (double* thisLayerError = errorInThisLayer.Elements(); thisLayerError != thisLayerErrorEnd; ++thisLayerError)
	  {
		error += (*weight * *thisLayerError);
		weight += _weights->Columns();
	  }
	  *prevLayerError = error;
	  ++weightColumnStart;
	}
  }
}

void FullyConnectedLayer::UpdateWeightAndBiasErrors(const Tensor& delta, const Tensor& previousLayerActivations, Tensor& nablaW, Tensor& nablaB,
  const DropoutMask* dropoutMask)
{
#ifdef _DEBUG
  if (nablaW.Size() != _weights->Rows() * _weights->Columns())
	throw std::runtime_error("FullyConnectedLayer::UpdateWeightAndBiasErrors - Size of nablaW does not match the number of weights.");
  if (nablaB.Size() != _weights->Rows())
	throw std::runtime_error("FullyConnectedLayer::UpdateWeightAndBiasErrors - Size of nablaB does not match the number of biases.");
#endif
  // Do a vector multiplication of delta by the transpose of previousLayerActivations
  // and store the result in nablaW.
  double* result = nablaW.Elements();
  double* e1 = delta.Elements();
  if (dropoutMask)
  {
	double* nb = nablaB.Elements();
	const bool* keep = dropoutMask->Begin();
	for (size_t r = 0; r < delta.Size(); ++r)
	{
	  if (*keep)
	  {
		*nb += *e1;
		double* e2 = previousLayerActivations.Elements();
		for (size_t c = 0; c < previousLayerActivations.Size(); ++c)
		{
		  *result += (*e1 * *e2);
		  ++e2;
		  ++result;
		}
	  }
	  else
	  {
		result += previousLayerActivations.Size();
	  }
	  ++e1;
	  ++nb;
	  ++keep;
	}
  }
  else
  {
    nablaB.ComponentWiseAdd(delta);
	for (size_t r = 0; r < delta.Size(); ++r)
	{
	  double* e2 = previousLayerActivations.Elements();
	  for (size_t c = 0; c < previousLayerActivations.Size(); ++c)
	  {
		*result += (*e1 * *e2);
		++e2;
		++result;
	  }
	  ++e1;
	}
  }
}

void FullyConnectedLayer::SwitchToTrainingWeights()
{
  if (_trainingWeights)
	memcpy(_weights->Elements(), _trainingWeights->Elements(), sizeof(double) * _weights->Size());
}

void FullyConnectedLayer::SwitchToTestingWeights()
{
  if (_trainingWeights)
  {
	// First save the training weights.
	memcpy(_trainingWeights->Elements(), _weights->Elements(), sizeof(double) * _weights->Size());
	// Now scale all the weights to compensate for the absence dropout.
	const double* end = _weights->Elements() + _weights->Size();
	double scale = static_cast<double>(_prevLayerKeepProbability);
	for (double* w = _weights->Elements(); w < end; ++w)
	  *w *= scale;
  }
}

MaxPoolingLayer::MaxPoolingLayer(uint32_t inputChannelCount, uint32_t inputRows, uint32_t inputColumns)
  : Layer(inputChannelCount, inputRows / 2, inputColumns / 2),
	_inputChannelCount(inputChannelCount), _inputRows(inputRows), _inputColumns(inputColumns)
{
  if (inputColumns % 2 != 0 || inputRows % 2 != 0)
	throw std::runtime_error("Input dimensions to MaxPoolingLayer must be divisible by 2.");
}

void MaxPoolingLayer::Description(std::ostream& os) const
{
  os << "Max pooling 2 by 2, input dimensions: " << _inputChannelCount << 'x' << _inputColumns << 'x' << _inputRows;
}

void MaxPoolingLayer::FeedForward(const Tensor& inputs, Tensor& outputs, const DropoutMask*) const
{
#ifdef _DEBUG
  if (inputs.Planes() != _inputChannelCount)
	throw std::runtime_error("MaxPoolingLayer::FeedForward - input tensor has the wrong number of channels.");
  if (inputs.Rows() != _inputRows)
	throw std::runtime_error("MaxPoolingLayer::FeedForward - input tensor has the wrong number of rows.");
  if (inputs.Columns() != _inputColumns)
	throw std::runtime_error("MaxPoolingLayer::FeedForward - input tensor has the wrong number of columns.");
  if (outputs.Planes() != _inputChannelCount)
	throw std::runtime_error("MaxPoolingLayer::FeedForward - output tensor has the wrong number of channels.");
  if (outputs.Rows() != _inputRows / 2)
	throw std::runtime_error("MaxPoolingLayer::FeedForward - output tensor has the wrong number of rows.");
  if (outputs.Columns() != _outputColumns)
	throw std::runtime_error("MaxPoolingLayer::FeedForward - output tensor has the wrong number of columns.");
#endif
  const double* row1 = inputs.Elements();
  const double* row2 = row1 + _inputColumns;
  const double* outputEnd = outputs.Elements() + outputs.Size();
  double* output = outputs.Elements();
  while (output < outputEnd)
  {
	const double* outputRowEnd = output + _outputColumns;
	while (output < outputRowEnd)
	{
	  double out = *row1;
	  ++row1;
	  if (*row1 > out)
		out = *row1;
	  ++row1;
	  if (*row2 > out)
		out = *row2;
	  ++row2;
	  if (*row2 > out)
		out = *row2;
	  ++row2;
	  *output = out;
	  ++output;
	}
	row1 += _inputColumns;
	row2 += _inputColumns;
  }
}

void MaxPoolingLayer::Save(std::ofstream& os) const
{
  os.put((char)Types::MaxPooling);
}

void MaxPoolingLayer::SaveArchitecture(std::ostream& os) const
{
  os << "Max pooling" << std::endl;
}

void MaxPoolingLayer::BackpropagateError(const Tensor& thisLayerActivations, const Tensor& previousLayerActivations,
  const Tensor& errorInThisLayer, Tensor& errorInPreviousLayer) const
{
#ifdef _DEBUG
  if (previousLayerActivations.Planes() != _inputChannelCount)
	throw std::runtime_error("MaxPoolingLayer::FeedForward - previous layer activations tensor has the wrong number of channels.");
  if (previousLayerActivations.Rows() != _inputRows)
	throw std::runtime_error("MaxPoolingLayer::FeedForward - previous layer activations tensor has the wrong number of rows.");
  if (previousLayerActivations.Columns() != _inputColumns)
	throw std::runtime_error("MaxPoolingLayer::FeedForward - previous layer activations tensor has the wrong number of columns.");
  if (thisLayerActivations.Planes() != _inputChannelCount)
	throw std::runtime_error("MaxPoolingLayer::FeedForward - this layer activations tensor has the wrong number of channels.");
  if (thisLayerActivations.Rows() != _inputRows / 2)
	throw std::runtime_error("MaxPoolingLayer::FeedForward - this layer activations tensor has the wrong number of rows.");
  if (thisLayerActivations.Columns() != _outputColumns)
	throw std::runtime_error("MaxPoolingLayer::FeedForward - this layer activations tensor has the wrong number of columns.");

  if (errorInPreviousLayer.Planes() != _inputChannelCount)
	throw std::runtime_error("MaxPoolingLayer::BackpropagateError - error in previous layer tensor has the wrong number of channels.");
  if (errorInPreviousLayer.Rows() != _inputRows)
	throw std::runtime_error("MaxPoolingLayer::BackpropagateError - error in previous layer tensor has the wrong number of rows.");
  if (errorInPreviousLayer.Columns() != _inputColumns)
	throw std::runtime_error("MaxPoolingLayer::BackpropagateError - error in previous layer tensor has the wrong number of columns.");
  if (errorInThisLayer.Planes() != _inputChannelCount)
	throw std::runtime_error("MaxPoolingLayer::BackpropagateError - error in this layer tensor has the wrong number of channels.");
  if (errorInThisLayer.Rows() != _inputRows / 2)
	throw std::runtime_error("MaxPoolingLayer::BackpropagateError - error in this layer tensor has the wrong number of rows.");
  if (errorInThisLayer.Columns() != _outputColumns)
	throw std::runtime_error("MaxPoolingLayer::BackpropagateError - error in this layer tensor has the wrong number of columns.");
#endif
  const double* prevLayerActivationRow1 = previousLayerActivations.Elements();
  const double* prevLayerActivationRow2 = prevLayerActivationRow1 + _inputColumns;
  double* prevLayerErrorRow1 = errorInPreviousLayer.Elements();
  double* prevLayerErrorRow2 = prevLayerErrorRow1 + _inputColumns;
  const double* thisLayerError = errorInThisLayer.Elements();
  const double* thisLayerActivationEnd = thisLayerActivations.Elements() + thisLayerActivations.Size();
  const double* thisLayerActivation = thisLayerActivations.Elements();
  while (thisLayerActivation < thisLayerActivationEnd)
  {
	const double* thisLayerActivationRowEnd = thisLayerActivation + _outputColumns;
	while (thisLayerActivation < thisLayerActivationRowEnd)
	{
	  double activation = *thisLayerActivation;
	  // Set even row errors.
	  *prevLayerErrorRow1 = (*prevLayerActivationRow1 == activation ? *thisLayerError : 0.0);
	  ++prevLayerActivationRow1;
	  ++prevLayerErrorRow1;
	  *prevLayerErrorRow1 = (*prevLayerActivationRow1 == activation ? *thisLayerError : 0.0);
	  ++prevLayerActivationRow1;
	  ++prevLayerErrorRow1;
	  // Set odd row errors.
	  *prevLayerErrorRow2 = (*prevLayerActivationRow2 == activation ? *thisLayerError : 0.0);
	  ++prevLayerActivationRow2;
	  ++prevLayerErrorRow2;
	  *prevLayerErrorRow2 = (*prevLayerActivationRow2 == activation ? *thisLayerError : 0.0);
	  ++prevLayerActivationRow2;
	  ++prevLayerErrorRow2;
	  ++thisLayerActivation;
	  ++thisLayerError;
	}
	prevLayerActivationRow1 += _inputColumns;
	prevLayerActivationRow2 += _inputColumns;
	prevLayerErrorRow1 += _inputColumns;
	prevLayerErrorRow2 += _inputColumns;
  }
}
