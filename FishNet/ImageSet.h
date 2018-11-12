#pragma once

#include "Image.h"

class ImageSet
{
public:
  ImageSet(const std::string& name, std::vector<std::string>&& categories, uint32_t channels, uint32_t width, uint32_t height);
  ~ImageSet();
  void ReserveTrainingSpace(uint32_t trainingSetSize)
  {
	_trainingSet.reserve(trainingSetSize);
  }
  void ReserveTestSpace(uint32_t testSetSize)
  {
    _testSet.reserve(testSetSize);
  }
  void AddImage(Image& image, bool isTest)
  {
	if (isTest)
	  _testSet.push_back(&image);
	else
	  _trainingSet.push_back(&image);
  }
  const std::string& Name() const noexcept { return _name; }
  const std::vector<Image*>& TrainingSet() const noexcept { return _trainingSet; }
  const std::vector<Image*>& TestSet() const noexcept { return _testSet; }
  const std::vector<Tensor>& OneHotCategories() const noexcept { return _oneHotCategories; }
  const std::vector<std::string>& Categories() const noexcept { return _categories; }
  uint32_t Channels() const { return _channels; }
  uint32_t Width() const { return _width; }
  uint32_t Height() const { return _height; }
private:
  ImageSet(const ImageSet&) = delete;

  std::string _name;
  std::vector<Image*> _trainingSet;
  std::vector<Image*> _testSet;
  std::vector<std::string> _categories;
  std::vector<Tensor> _oneHotCategories;
  uint32_t _channels;
  uint32_t _width;
  uint32_t _height;
};
