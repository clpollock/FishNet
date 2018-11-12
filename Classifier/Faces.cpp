#include "stdafx.h"
#include "PGM.h"
#include <ImageSet.h>

enum class FaceClassifications { Emotions, Directions, People, Sunglasses, DirectionsSunglasses };

static std::vector<std::string> directions = { "left", "right", "straight", "up" };

static void LoadFaces(ImageSet& imageSet, const std::string& dataDir, const std::string& listFile,
  const std::string& suffix, int expectedWidth, int expectedHeight, FaceClassifications classification,
  bool isTestSet)
{
  std::string path = dataDir + listFile;
  std::ifstream is(path);
  if (!is.good())
	throw std::runtime_error("Failed to open file " + path);
  std::string imageDir = dataDir + "faces/";
  // Inelegant, but it works, and time is short.
  std::string line;
  int lineNo = 0;
  while (std::getline(is, line))
  {
	++lineNo;
	// Determine the label from the file name. This isn't pretty, but it works reliably
	// and avoids the tedious and error-prone task of categorizing the images manually.
	int label = -1;
	if (classification == FaceClassifications::Sunglasses)
	{
	  label = line.find("sunglasses") == std::string::npos ? 0 : 1;
	}
	else
	{
	  const auto& categories(classification == FaceClassifications::DirectionsSunglasses ?
		directions : imageSet.Categories());
	  // The file name should include the label.
	  for (int i = 0; i < categories.size(); ++i)
	  {
		if (line.find(categories[i]) != std::string::npos)
		{
		  if (label != -1)
		  {
			std::stringstream ss;
			ss << "Image " << line << " at line " << lineNo << " of " << path << " belongs to 2 categories: "
			  << categories[label] << " and " << categories[i] << std::endl;
			throw std::runtime_error(ss.str());
		  }
		  label = i;
		}
	  }

	  if (classification == FaceClassifications::DirectionsSunglasses && line.find("sunglasses") != std::string::npos)
		label += 4;
	}

	if (label == -1)
	{
	  std::stringstream ss;
		ss << "Image " << line << " at line " << lineNo << " of " << path
		  << " does not match any category." << std::endl;
		throw std::runtime_error(ss.str());
	}
	std::string fileName(imageDir + line + suffix);
	Image& image = LoadPGMImage(fileName, label);

	// Output the image as ASCII art to verify that the loader is working correctly.
//	std::cout << line << ", " << label << std::endl;
//	const char* greyscale = "@%#*+=-:. ";
//	for (int row = 0; row < expectedHeight; ++row)
//	{
//	  for (int col = 0; col < expectedWidth; ++col)
//	  {
//		double pixel = image.Inputs().Get(0, row, col);
//		int k = floor(pixel * 10.0);
//		std::cout << greyscale[k];
//	  }
//	  std::cout << std::endl;
//	}

	if (image.Inputs().Columns() != expectedWidth)
	{
	  std::stringstream msg;
	  msg << "Width of " << fileName << " (" << image.Inputs().Columns()
		<< ") does not match expected width of " << expectedWidth;
	  throw std::runtime_error(msg.str());
	}
	if (image.Inputs().Rows() != expectedHeight)
	{
	  std::stringstream msg;
	  msg << "Height of " << fileName << " (" << image.Inputs().Rows()
		<< ") does not match expected height of " << expectedHeight;
	  throw std::runtime_error(msg.str());
	}
	imageSet.AddImage(image, isTestSet);
  }
}

std::unique_ptr<ImageSet> LoadFaces(const std::string& dataDir, const std::string& classification, bool dry)
{
  int size = std::stoi(Utils::GetEnv("FISHNET_FACE_IMAGE_SIZE"));
  std::string suffix;
  switch (size)
  {
	case 1:
	  suffix = ".pgm";
	  break;
	case 2:
	  suffix = "_2.pgm";
	  break;
	case 4:
	  suffix = "_4.pgm";
	  break;
	default:
	  throw std::runtime_error("Environment variable FACE_IMAGE_SIZE has invalid value. It must be 1, 2, or 4.");
  }
  int width = 128 / size;
  int height = 120 / size;
  std::vector<std::string> categories;
  FaceClassifications fc;
  if (classification == "emotions")
  {
	fc = FaceClassifications::Emotions;
	categories = { "angry", "happy", "neutral", "sad" };
  }
  else if (classification == "face-directions")
  {
	fc = FaceClassifications::Directions;
	categories = directions;
  }
  else if (classification == "people")
  {
	fc = FaceClassifications::People;
	categories = { "an2i", "at33", "boland", "bpm", "ch4f", "cheyer", "choon", "danieln", "glickman", "karyadi",
	  "kawamura", "kk49", "megak", "mitchell", "night", "phoebe", "saavik", "steffi", "sz24", "tammo" };
  }
  else if (classification == "sunglasses")
  {
	fc = FaceClassifications::Sunglasses;
	categories = { "no sunglasses", "sunglasses" };
  }
  else if (classification == "directions-sunglasses")
  {
	fc = FaceClassifications::DirectionsSunglasses;
	categories = { "left", "right", "straight", "up",
	  "left sunglasses", "right sunglasses", "straight sunglasses", "up sunglasses" };
  }
  else
  {
	throw std::runtime_error("Unknown face classification:" + classification);
  }

  std::stringstream ss;
  ss << classification << " (" << width << 'x' << height << ')';
  std::unique_ptr<ImageSet> imageSet = std::make_unique<ImageSet>(ss.str(), std::move(categories), 1, width, height);
  if (!dry)
  {
	LoadFaces(*imageSet, dataDir, "training_faces.txt", suffix, width, height, fc, false);
	LoadFaces(*imageSet, dataDir, "test_faces.txt", suffix, width, height, fc, true);
  }
  return imageSet;
}
