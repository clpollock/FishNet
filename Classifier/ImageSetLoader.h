#pragma once

#include <ImageSet.h>

class ImageSetLoader
{
public:
  ImageSetLoader(bool dry);
  const ImageSet& Load(const std::string& name);
private:
  ImageSetLoader(const ImageSetLoader&) = delete;

  std::map<std::string, std::unique_ptr<ImageSet>> _imageSets;
  std::string _dataDir;
  bool _dry;
};
