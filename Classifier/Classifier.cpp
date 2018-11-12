#include "stdafx.h"
#include "MNIST.h"
#include "CIFAR.h"
#include "Faces.h"
#include "Trainer.h"

static void TestNetwork(FeedForwardNetwork& network, const ImageSet& imageSet, std::ostream& os)
{
  os << "Dataset," << imageSet.Name() << std::endl
	<< "Training epochs," << network.EpochsTrained() << std::endl << std::endl;
  network.SaveArchitecture(os);
  os << std::endl;
  network.SaveWeightStatistics(os);
  os << std::endl << "Network Classifications" << std::endl;
  // Test network and save statistics.
  LOG(Info) << "Network architecture:" << std::endl << network;
  LOG(Info) << "Testing on " << imageSet.Name() << " dataset.";
  LOG(Info) << "Using " << network.ThreadCount() << " threads.";
  auto testingStart = std::chrono::steady_clock::now();
  network.SaveAccuracyStatistics(imageSet, os);
  auto testingEnd = std::chrono::steady_clock::now();
  LOG(Info) << "Testing completed in "
	<< std::chrono::duration_cast<std::chrono::milliseconds>(testingEnd - testingStart).count() << " ms." << std::endl;
}

int main(int argc, char* argv[])
{
  uint32_t threadCount = std::thread::hardware_concurrency();
  std::string dataSet;
  std::vector<std::string> files;
  std::string outputFile;
  bool dry = false;
  bool test = false;

  try
  {
	for (int ai = 1; ai < argc; ++ai)
	{
	  if (argv[ai][0] != '-')
	  {
		files.emplace_back(argv[ai]);
	  }
	  else
	  {
		std::string arg = argv[ai];
		StringUtils::ToLower(arg);
		if (arg == "-dataset")
		{
		  if (++ai == argc)
			throw std::runtime_error("-dataset must be followed by the data set name.");
		  dataSet = argv[ai];
		  StringUtils::ToLower(dataSet);
		}
		else if (arg == "-dry")
		{
		  dry = true;
		}
		else if (arg == "-output")
		{
		  if (++ai == argc)
			throw std::runtime_error("-output must be followed by a file name.");
		  outputFile = argv[ai];
		}
		else if (arg == "-test")
		{
		  test = true;
		}
		else if (arg == "-threads")
		{
		  if (++ai == argc)
			throw std::runtime_error("-threads must be followed by the number of threads.");
		  threadCount = std::atoi(argv[ai]);
		  if (threadCount < 1)
			throw std::runtime_error("Must use at least 1 thread.");
		}
		else
		{
		  throw std::runtime_error("Unrecognized argument: " + arg);
		}
	  }
	}
  }
  catch (const std::exception& e)
  {
	std::cerr << e.what() << std::endl;
	return 1;
  }

  if (test)
  {
	if (dry)
	{
	  std::cerr << "-dry cannot be used with -test" << std::endl;
	  return 1;
	}
  }
  else
  {
	if (!dataSet.empty())
	{
	  std::cerr << "-dataset can only be used with -test" << std::endl;
	  return 1;
	}
	if (!outputFile.empty())
	{
	  std::cerr << "-output can only be used with -test" << std::endl;
	  return 1;
	}
  }

  if (files.empty())
  {
	std::cerr << "You need to supply the name of at least one file." << std::endl;
	return 1;
  }
  
  std::unique_ptr<LogAdaptor> logAdaptor;
  try
  {
	logAdaptor.reset(new LogFileAdaptor(Utils::GetEnv("FISHNET_LOG_DIR"), "Classifier"));
	ImageSetLoader imageSetLoader(dry);
	if (test)
	{
	  if (dataSet.empty())
	  {
		std::cerr << "You need to supply the name of the dataset to test with." << std::endl;
		return 1;
	  }
	  const ImageSet& imageSet = imageSetLoader.Load(dataSet);
	  // Save to a file if the user supplied a file name, otherwise output to std::cout.
	  std::ofstream ofs;
	  if (!outputFile.empty())
	  {
		ofs.open(outputFile.c_str(), std::ofstream::trunc);
		if (!ofs.good())
		  throw std::runtime_error("Failed to open file " + outputFile + " for writing.");
		LOG(Info) << "Saving statistics to " << outputFile;
	  }
	  std::ostream& os = outputFile.empty() ? std::cout : ofs;
	  for (const std::string& file : files)
	  {
		std::unique_ptr<FeedForwardNetwork> network = FeedForwardNetwork::Load(file, threadCount);
		TestNetwork(*network, imageSet, os);
		os << std::endl;
	  }
	}
	else
	{
	  Trainer trainer(imageSetLoader, threadCount);
	  trainer.LoadJobList(files);
	  LOG(Info) << "Loaded the following " << trainer.Jobs().size() << " training jobs: " << std::endl << trainer.Jobs();
	  if (!dry)
	  {
		trainer.TrainAll();
	  }
	}
  }
  catch (const std::exception& e)
  {
	LOG(Error) << e.what();
	std::cerr << e.what() << std::endl;
	return 1;
  }
  return 0;
}
