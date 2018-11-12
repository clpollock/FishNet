#include "stdafx.h"
#include "CIFAR.h"

void LoadCIFARData(ImageSet& imageSet, const std::string& imageFilename, bool isTestSet)
{
  // OpenFile will throw an exception if the file cannot be opened.
  FILE* imageFile = Utils::OpenFile(imageFilename, "rb");
  // Each image consists of a 1 byte label and 3072 bytes of pixel data.
  const uint32_t imageSize = 3 * 32 * 32;
  const uint32_t recordSize = 1 + imageSize;
  auto buffer = std::make_unique<unsigned char[]>(recordSize * 10000);
  unsigned char* data = buffer.get();
  fread(data, 10000, recordSize, imageFile);
 
  for (uint32_t i = 0; i < 10000; ++i)
  {
	uint32_t label = *data;
	if (label > 9)
	{
	  fclose(imageFile);
	  throw std::runtime_error("Invalid label.");
	}
	++data;
	auto vectorData = std::make_unique<double[]>(imageSize);
	for (uint32_t j = 0; j < imageSize; ++j)
	{
	  vectorData[j] = double(*data) / 255.0;
	  ++data;
	}
	auto& image = *new Image(move(vectorData), 3, 32, 32, label);
	imageSet.AddImage(image, isTestSet);
  }

  fclose(imageFile);
}

extern std::unique_ptr<ImageSet> LoadCIFAR10(const std::string& dataDir, bool dry)
{
  std::string cifarDir = dataDir + "cifar-10-batches-bin/";
  // Load category labels.
  std::vector<std::string> categories(10);
  if (!dry)
  {
	std::string labelName(cifarDir + "batches.meta.txt");
	std::ifstream is(labelName);
	if (!is.good())
	  throw std::runtime_error("Failed to open file " + labelName);
	for (int category = 0; category < 10; ++category)
	  std::getline(is, categories[category]);
	is.close();
  }
  std::unique_ptr<ImageSet> imageSet = std::make_unique<ImageSet>("CIFAR-10", std::move(categories), 3, 32, 32);
  if (!dry)
  {
	imageSet->ReserveTrainingSpace(50000);
	imageSet->ReserveTestSpace(10000);
	for (char c = '1'; c < '6'; ++c)
	{
	  std::string fileName = cifarDir + "data_batch_" + c + ".bin";
	  LoadCIFARData(*imageSet, fileName, false);
	}

	LoadCIFARData(*imageSet, cifarDir + "test_batch.bin", true);
  }
  return imageSet;
}
