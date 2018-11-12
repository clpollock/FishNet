#pragma once

#include "Layer.h"

class ConvolutionalLayer : public WeightedLayer
{
public:
  // The weights Tensor is 4 dimensional.
  // The dimensions are weight row, weight column, input channel, and filter (output channel).
  // The bias Tensor is one-dimensional, with one bias for each filter.
  ConvolutionalLayer(TensorPtr&& weights, TensorPtr&& biases, uint32_t inputRows, uint32_t inputColumns, uint32_t stride,
	uint32_t zeroPadding, std::unique_ptr<::ActivationFunction>&&);
  ConvolutionalLayer(uint32_t inputChannelCount, uint32_t inputRows, uint32_t inputColumns, uint32_t filterCount,
	uint32_t filterSize, uint32_t stride, uint32_t zeroPadding, std::unique_ptr<::ActivationFunction>&&);
  ~ConvolutionalLayer() {}
  virtual void InitializeWeights() override;
  virtual void Description(std::ostream&) const override;
  virtual void Save(std::ofstream&) const override;
  virtual void SaveArchitecture(std::ostream&) const override;
  virtual void FeedForward(const Tensor& inputs, Tensor& outputs, const DropoutMask*) const override;
  virtual void BackpropagateError(const Tensor& errorInThisLayer, Tensor& errorInPreviousLayer, const DropoutMask*) const override;
  virtual void UpdateWeightAndBiasErrors(const Tensor& delta, const Tensor& previousLayerActivations,
	Tensor& nablaW, Tensor& nablaB, const DropoutMask*) override;
private:
  struct FilterInfo
  {
	int inputStartOffset;
	int outputStartOffset;
	int outputSpan;
  };

  void CalculateFilterInfo();
  void CalculateFilterInfo(ConvolutionalLayer::FilterInfo* filterInfo, int32_t inputDimensionLength);

  std::unique_ptr<FilterInfo[]> _filterRowInfo;
  std::unique_ptr<FilterInfo[]> _filterColumnInfo;
  uint32_t _inputChannelCount;
  int32_t _inputRows;
  int32_t _inputColumns;
  uint32_t _filterCount;
  int32_t _filterSize;
  int32_t  _stride;
  int32_t  _zeroPadding;
};
