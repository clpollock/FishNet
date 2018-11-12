#pragma once

#include "ImageSetLoader.h"
#include <FeedForwardNetwork.h>

class Job
{
public:
  Job(const ImageSet& dataSet, std::unique_ptr<FeedForwardNetwork> network, uint32_t epochs, uint32_t giveUpAfter,
	uint32_t miniBatchSize, double learningRateDecay, double learningRateDecayPoint)
	: _dataSet(dataSet), _network(move(network)),
	  _learningRateDecay(learningRateDecay), _learningRateDecayPoint(learningRateDecayPoint),
  	  _epochs(epochs), _giveUpAfter(giveUpAfter), _miniBatchSize(miniBatchSize) {}
  ~Job() {}
  void Run(const std::string& saveDir)
  {
	_network->Train(_dataSet, _epochs, _giveUpAfter, _miniBatchSize, _learningRateDecay, _learningRateDecayPoint, saveDir);
  }
  const ImageSet& DataSet() const { return _dataSet; }
  const FeedForwardNetwork& Network() const { return *_network; }
  uint32_t Epochs() const { return _epochs; }
  uint32_t GiveUpAfter() const { return _giveUpAfter; }
  uint32_t MiniBatchSize() const { return _miniBatchSize; }
  double LearningRateDecay() const { return _learningRateDecay; }
  double LearningRateDecayPoint() const { return _learningRateDecayPoint; }
private:
  const ImageSet& _dataSet;
  std::unique_ptr<FeedForwardNetwork> _network;
  double _learningRateDecay;
  double _learningRateDecayPoint;
  uint32_t _epochs;
  uint32_t _giveUpAfter;
  uint32_t _miniBatchSize;
};

std::ostream& operator<<(std::ostream&, const Job&);
std::ostream& operator<<(std::ostream&, const std::vector<std::unique_ptr<Job>>&);

class Trainer
{
public:
  Trainer(ImageSetLoader& imageSetLoader, uint32_t threadCount)
	: _imageSetLoader(imageSetLoader),
	  _threadCount(threadCount) {}
  void LoadJobList(std::vector<std::string>& jobFiles)
  {
	for (const std::string& fileName : jobFiles)
	  LoadJobList(fileName);
  }
  void LoadJobList(const std::string& fileName);
  void TrainAll();
  const std::vector<std::unique_ptr<Job>>& Jobs() const { return _jobs; }
  uint32_t ThreadCount() const { return _threadCount;  }
private:
  std::unique_ptr<FeedForwardNetwork> LoadNetwork(const std::string& name, std::ifstream&,
	int& lineNo, const ImageSet& imageSet, double learningRate, double weightDecay);

  ImageSetLoader& _imageSetLoader;
  std::vector<std::unique_ptr<Job>> _jobs;
  uint32_t _threadCount;
};
