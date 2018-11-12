#include "stdafx.h"
#include "ImageSet.h"

ImageSet::ImageSet(const std::string& name, std::vector<std::string>&& categories, uint32_t channels, uint32_t width, uint32_t height)
  : _name(name), _categories(std::move(categories)), _channels(channels), _width(width), _height(height)
{
  // Our target activation for each category is 1 for the neuron corresponding to the category and
  // 0 for all other neurons.
  uint32_t numberOfCategories = static_cast<uint32_t>(_categories.size());
  for (uint32_t category = 0; category < numberOfCategories; ++category)
  {
	_oneHotCategories.emplace_back(numberOfCategories);
	_oneHotCategories.back().Set(category, 1.0);
  }
}

ImageSet::~ImageSet()
{
  for (Image* i : _trainingSet)
	delete i;
  for (Image* i : _testSet)
	delete i;
}
