#include "stdafx.h"
#include <Image.h>
#include <Tokenizer.h>

Image& LoadPGMImage(const std::string& fileName, uint32_t label)
{
  std::ifstream stream(fileName, std::ifstream::in|std::ios::binary|std::ios::ate);
  if (!stream.good())
	throw std::runtime_error("Could not open file " + fileName);
  std::streampos size = stream.tellg();
  stream.seekg(0);
  auto buffer = std::make_unique<char[]>(size);
  stream.read(buffer.get(), size);
  stream.close();

  Tokenizer tokenizer(buffer.get(), size);
  std::string magic = tokenizer.NextToken();
  if (magic != "P2" && magic != "P5")
  {
	std::stringstream msg;
	msg << "Unrecognized magic number " << magic << " in " << fileName;
	throw std::runtime_error(msg.str());
  }
  uint32_t width = std::stoi(tokenizer.NextToken());
  uint32_t height = std::stoi(tokenizer.NextToken());
  double maxVal = std::stod(tokenizer.NextToken());
  if (maxVal < 1.0 || maxVal > 65535.0)
  {
	std::stringstream msg;
	msg << fileName + " has invalid Maxval field: " << maxVal;
	throw std::runtime_error(msg.str());
  }
  int pixelBytes(maxVal < 256.0 ? 1 : 2);
  uint32_t pixelCount = width * height;
  auto vectorData = std::make_unique<double[]>(pixelCount);
  if (magic == "P5")
  {
	// Skip 1 byte of whitespace.
	const char* data = tokenizer.Position() + 1;
	int64_t remainingBytes = (buffer.get() + size) - data;
	int64_t missing = pixelCount * pixelBytes - remainingBytes;
	if (missing > 0)
	{
	  LOG(Warning) << fileName << " appears to be missing " << missing << " bytes.";
	}
	if (pixelBytes == 1)
	{
	  for (uint32_t i = 0; i < pixelCount; ++i)
	  {
		vectorData[i] = double(*data) / maxVal;
		++data;
	  }
	}
	else
	{
	  for (uint32_t i = 0; i < pixelCount; ++i)
	  {
		uint32_t pixel = *data;
		++data;
		pixel <<= 8;
		pixel |= *data;
		++data;
		vectorData[i] = double(pixel) / maxVal;
	  }
	}
  }
  else
  {
	for (uint32_t i = 0; i < pixelCount; ++i)
	{
	  std::string p = tokenizer.NextToken();
	  if (p.empty())
		throw std::runtime_error(fileName + " appears to be incomplete.");
	  double pixel = std::stod(p);
	  if (pixel < 0.0 || pixel > maxVal)
		throw std::runtime_error(fileName + " contains invalid pixel data.");
	  vectorData[i] = pixel / maxVal;
	}
  }
  return *new Image(move(vectorData), 1, width, height, label);
}
