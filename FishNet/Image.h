#pragma once

#include "Tensor.h"

class Image
{
  public:
	Image(std::unique_ptr<double[]>&& data, uint32_t channels, uint32_t width, uint32_t height, uint32_t category)
	  : _inputs(move(data), channels, height, width), _category(category) {}
	Tensor& Inputs() { return _inputs; }
	const Tensor& Inputs() const { return _inputs; }
	uint32_t Category() const { return _category; }
  private:
	Tensor _inputs;
	uint32_t _category;
};
