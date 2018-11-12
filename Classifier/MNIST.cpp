#include "stdafx.h"
#include "MNIST.h"

static uint32_t ReadInt(FILE* file)
{
  unsigned char v[4];
  fread(v, 1, 4, file);
  // The file has the bytes in big-endian order so we have to change to little-endian to run on x86 processors.
  uint32_t ret = v[0];
  ret <<= 8;
  ret |= (unsigned char)v[1];
  ret <<= 8;
  ret |= (unsigned char)v[2];
  ret <<= 8;
  ret |= (unsigned char)v[3];
  return ret;
}

void LoadMNISTData(ImageSet& imageSet, const std::string& imageFilename, const std::string& labelFilename, bool isTestSet)
{
  FILE* imageFile = Utils::OpenFile(imageFilename, "rb");
  FILE *labelFile = Utils::OpenFile(labelFilename, "rb");

  if (ReadInt(imageFile) != 2051)
	throw std::runtime_error(imageFilename + " is not a valid MNIST image file.");
  if (ReadInt(labelFile) != 2049)
	throw std::runtime_error(labelFilename + " is not a valid MNIST label file.");

  uint32_t imageCount = ReadInt(imageFile);
  uint32_t labelCount = ReadInt(labelFile);

  if (imageCount != labelCount)
	throw std::runtime_error("Label count does not match image count.");

  if (isTestSet)
	imageSet.ReserveTestSpace(imageCount);
  else
	imageSet.ReserveTrainingSpace(imageCount);

  uint32_t imageHeight = ReadInt(imageFile);
  uint32_t imageWidth = ReadInt(imageFile);

  if (imageHeight != 28 || imageWidth != 28)
	throw std::runtime_error("Incorrect image size.");

  auto imageBuffer = std::make_unique<unsigned char[]>(imageCount * 28 * 28);
  fread(imageBuffer.get(), imageCount, 28 * 28, imageFile);

  auto labelBuffer = std::make_unique<unsigned char[]>(imageCount);
  fread(labelBuffer.get(), imageCount, 1, labelFile);

  unsigned char* data = imageBuffer.get();
  for (uint32_t i = 0; i < imageCount; ++i)
  {
	auto vectorData = std::make_unique<double[]>(28 * 28);
	for (uint32_t j = 0; j < 28 * 28; ++j)
	{
	  // Convert the pixel value into a double ranging from 0 to 1.
	  vectorData[j] = double(*data) / 255.0;
	  ++data;
	}

	auto& image = *new Image(move(vectorData), 1, 28, 28, uint32_t(labelBuffer[i]));
	imageSet.AddImage(image, isTestSet);
  }

  if (imageFile)
	fclose(imageFile);

  if (labelFile)
	fclose(labelFile);
}

extern std::unique_ptr<ImageSet> LoadMNIST(const std::string& dataDir, bool dry)
{
  std::string mnistDir = dataDir + "MNIST/";
  std::vector<std::string> categories;
  for (int n = 0; n < 10; ++n)
	categories.emplace_back(std::to_string(n));
  std::unique_ptr<ImageSet> imageSet = std::make_unique<ImageSet>("MNIST", std::move(categories), 1, 28, 28);
  if (!dry)
  {
	LoadMNISTData(*imageSet, mnistDir + "train-images-idx3-ubyte", mnistDir + "train-labels-idx1-ubyte", false);
	LoadMNISTData(*imageSet, mnistDir + "t10k-images-idx3-ubyte", mnistDir + "t10k-labels-idx1-ubyte", true);
  }
  return imageSet;
}
