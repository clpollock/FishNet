#pragma once

#include "ActivationFunction.h"
#include "Tensor.h"

class DropoutMask;

class Layer
{
public:
  enum class Types { FullyConnected = 0, Convolutional = 1, MaxPooling = 2 };

  virtual ~Layer() {}
  virtual void InitializeWeights() {}
  uint32_t OutputPlanes() const { return _outputPlanes; }
  uint32_t OutputRows() const { return _outputRows; }
  uint32_t OutputColumns() const { return _outputColumns; }
  virtual void Description(std::ostream&) const = 0;
  virtual double KeepProbability() const { return 1.0; }
  virtual void Save(std::ofstream&) const = 0;
  virtual void SaveArchitecture(std::ostream&) const = 0;
  static std::unique_ptr<Layer> Load(std::ifstream&, uint32_t inputChannelCount, uint32_t inputRows, uint32_t inputColumns,
	double prevLayerKeepProbability);
  virtual void FeedForward(const Tensor& inputs, Tensor& outputs, const DropoutMask*) const = 0;
  virtual void SwitchToTrainingWeights() {}
  virtual void SwitchToTestingWeights() {}
protected:
  Layer(uint32_t outputPlanes, uint32_t outputRows, uint32_t outputColumns)
	: _outputPlanes(outputPlanes), _outputRows(outputRows), _outputColumns(outputColumns) {}
  uint32_t _outputPlanes;
  uint32_t _outputRows;
  uint32_t _outputColumns;
};

class WeightedLayer : public Layer
{
public:
  virtual void BackpropagateError(const Tensor& errorInThisLayer, Tensor& errorInPreviousLayer, const DropoutMask*) const = 0;
  virtual void UpdateWeightAndBiasErrors(const Tensor& delta, const Tensor& previousLayerActivations,
	Tensor& nablaW, Tensor& nablaB, const DropoutMask*) = 0;
  void UpdateWeightsAndBiases(const Tensor& nablaW, const Tensor& nablaB, double scalar);
  void DecayWeights(double factor);
  void ApplyActivationFunction(Tensor& activations) const
  {
	if (_activationFunction)
	  _activationFunction->Apply(activations);
  }
  const Tensor& Weights() const { return *_weights; }
  const Tensor& Biases() const { return *_biases; }
  const ::ActivationFunction* ActivationFunction() const { return _activationFunction.get(); }
protected:
  WeightedLayer(TensorPtr&& weights, TensorPtr&& biases, std::unique_ptr<::ActivationFunction>&& activationFunction,
	uint32_t outputPlanes, uint32_t outputRows, uint32_t outputColumns)
	: Layer(outputPlanes, outputRows, outputColumns),
	  _weights(std::move(weights)), _biases(std::move(biases)), _activationFunction(std::move(activationFunction)) {}
  WeightedLayer(std::unique_ptr<::ActivationFunction>&& activationFunction,
	uint32_t outputPlanes, uint32_t outputRows, uint32_t outputColumns)
	: Layer(outputPlanes, outputRows, outputColumns),
	  _weights(nullptr), _biases(nullptr), _activationFunction(std::move(activationFunction)) {}

  TensorPtr _weights;
  TensorPtr _biases;
  std::unique_ptr<::ActivationFunction> _activationFunction;
};

class FullyConnectedLayer : public WeightedLayer
{
public:
  FullyConnectedLayer(TensorPtr&& weights, TensorPtr&& biases, std::unique_ptr<::ActivationFunction>&&,
	double keepProbability = 1.0, double prevLayerKeepProbability = 1.0);
  FullyConnectedLayer(uint32_t inputSize, uint32_t layerSize, std::unique_ptr<::ActivationFunction>&&,
	double keepProbability = 1.0, double prevLayerKeepProbability = 1.0);
  ~FullyConnectedLayer() {}
  virtual void InitializeWeights() override;
  virtual void Description(std::ostream&) const override;
  virtual double KeepProbability() const override { return _keepProbability; }
  virtual void Save(std::ofstream&) const override;
  virtual void SaveArchitecture(std::ostream&) const override;
  virtual void FeedForward(const Tensor& inputs, Tensor& outputs, const DropoutMask*) const override;
  virtual void BackpropagateError(const Tensor& errorInThisLayer, Tensor& errorInPreviousLayer, const DropoutMask*) const override;
  virtual void UpdateWeightAndBiasErrors(const Tensor& delta, const Tensor& previousLayerActivations,
	Tensor& nablaW, Tensor& nablaB, const DropoutMask*) override;
  virtual void SwitchToTrainingWeights() override;
  virtual void SwitchToTestingWeights() override;
private:
  // Used if the previous layer uses dropout
  TensorPtr _trainingWeights;
  double _prevLayerKeepProbability;
  double _keepProbability;
  uint32_t _inputSize;
};

// Hard code to 2 by 2 max pooling because this allows for a more efficient implementation.
// If I decide to try other pooling sizes I can easily implement them as separate classes.

class MaxPoolingLayer : public Layer
{
public:
  MaxPoolingLayer(uint32_t inputChannelCount, uint32_t inputRows, uint32_t inputColumns);
  virtual void Description(std::ostream&) const override;
  virtual void Save(std::ofstream&) const override;
  virtual void SaveArchitecture(std::ostream&) const override;
  virtual void FeedForward(const Tensor& inputs, Tensor& outputs, const DropoutMask*) const override;
  void BackpropagateError(const Tensor& thisLayerActivations, const Tensor& previousLayerActivations,
	const Tensor& errorInThisLayer, Tensor& errorInPreviousLayer) const;
private:
  uint32_t _inputChannelCount;
  uint32_t _inputRows;
  uint32_t _inputColumns;
};

class Randomizer
{
  public:
	Randomizer(double standardDeviation)
	  : _distribution(0, standardDeviation) {}
	double operator()() const
	{
	  return _distribution(_generator);
	}
	void Fill(Tensor&);
  private:
	static std::default_random_engine _generator;
	mutable std::normal_distribution<double> _distribution;
};
