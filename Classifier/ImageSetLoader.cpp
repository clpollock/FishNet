#include "stdafx.h"
#include "ImageSetLoader.h"
#include "CIFAR.h"
#include "MNIST.h"
#include "Faces.h"

ImageSetLoader::ImageSetLoader(bool dry)
  : _dry(dry)
{
  if (!dry)
  {
	_dataDir = Utils::GetEnv("FISHNET_DATA_DIR");
	if (_dataDir.back() != PATH_SEPARATOR)
	  _dataDir += PATH_SEPARATOR;
  }
}

const ImageSet& ImageSetLoader::Load(const std::string& name)
{
  auto i = _imageSets.find(name);
  if (i != _imageSets.end())
	return *i->second;
  std::unique_ptr<ImageSet> imageSet = nullptr;
  if (name == "cifar-10")
	imageSet = LoadCIFAR10(_dataDir, _dry);
  else if (name == "mnist")
	imageSet = LoadMNIST(_dataDir, _dry);
  else if (name == "emotions" || name == "face-directions" || name == "people" || name == "sunglasses" ||
	name == "directions-sunglasses")
	imageSet = LoadFaces(_dataDir, name, _dry);
  else
  {
	std::stringstream ss;
	ss << "Unknown image set " << name << ". The supported image sets are cifar-10, mnist, "
	  "emotions, face-directions, people, sunglasses, and directions-sunglasses.";
	throw std::runtime_error(ss.str());
  }
  auto result = _imageSets.emplace(name, std::move(imageSet));
  return *result.first->second;
}
