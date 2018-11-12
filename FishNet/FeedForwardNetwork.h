#pragma once

#include "DropoutMask.h"
#include "Layer.h"

class CostFunction;
class FeedForwardTrainer;
class Image;
class ImageSet;

class FeedForwardNetwork
{
public:
  using LayerVector = std::vector<std::unique_ptr<Layer>>;

  FeedForwardNetwork(const std::string& name, uint32_t inputChannelCount, uint32_t inputRows, uint32_t inputColumns,
	std::unique_ptr<::CostFunction>, uint32_t threadCount, uint16_t epochsTrained, double learningRate, double weightDecay);
  ~FeedForwardNetwork();
  void AddLayer(std::unique_ptr<Layer> layer)
  {
	_layers.emplace_back(move(layer));
  }
  void AddFullyConnectedLayer(uint32_t layerSize, std::unique_ptr<::ActivationFunction>&&, double keepProbability);
  void AddConvolutionalLayer(uint32_t filterCount, uint32_t filterSize, uint32_t stride, uint32_t zeroPadding,
	std::unique_ptr<::ActivationFunction>&&);
  void AddMaxPoolingLayer();
  void Save(const std::string& fileName);
  static std::unique_ptr<FeedForwardNetwork> Load(const std::string& fileName, uint32_t threadCount);
  const std::vector<Tensor>* OneHotCategories() const { return _oneHotCategories; }
  const LayerVector& Layers() const { return _layers; }
  const ::CostFunction& CostFunction() const { return *_costFunction;  }
  const Layer* TopLayer() const
  {
	return _layers.empty() ? nullptr : _layers.back().get();
  }
  uint32_t ThreadCount() const { return _threadCount; }
  uint32_t EpochsTrained() const { return _epochsTrained; }
  double LearningRate() const { return _learningRate; }
  double WeightDecay() const { return _weightDecay; }
  const std::string& Name() const { return _name; }
  void Name(const std::string& name)
  {
	_name = name;
  }
  void LearningRate(double rate)
  {
	_learningRate = rate;
  }
  void WeightDecay(double decay)
  {
	_weightDecay = decay;
  }
  std::vector<uint32_t> Classify(const ImageSet&);
  void SaveAccuracyStatistics(const ImageSet&, std::ostream&);
  void SaveWeightStatistics(std::ostream&) const;
  void SaveArchitecture(std::ostream&) const;
  void Train(const ImageSet&, uint32_t epochs, uint32_t giveUpAfter, uint32_t miniBatchSize,
	double learningRateDecay, double learningRateDecayPoint, const std::string& saveDir);
  void SignalWorkerFinished();
private:
  double TrainForOneEpoch(const std::vector<Image*>& trainingData, uint32_t miniBatchSize);
  std::pair<uint32_t, double> TestDuringTraining(const ImageSet&);
  void StartTrainers();
  void WaitForBackgroundTrainers()
  {
	std::unique_lock<std::mutex> lock(_mutex);
	if (_busyWorkerCount > 0)
	  _workersFinished.wait(lock, [this] { return _busyWorkerCount == 0; });
  }
  void StopTrainers();

  std::string _name;
  LayerVector _layers;
  std::unique_ptr<::CostFunction> _costFunction;
  uint32_t _inputChannelCount;
  uint32_t _inputRows;
  uint32_t _inputColumns;
  uint32_t _threadCount;
  uint32_t _epochsTrained;
  double _learningRate;
  double _weightDecay;
  double _weightDecayMultiplier;

  const std::vector<Tensor>* _oneHotCategories;

  std::vector<std::thread> _backgroundThreads;
  std::vector<std::unique_ptr<FeedForwardTrainer>> _backgroundTrainers;
  std::unique_ptr<FeedForwardTrainer> _foregroundTrainer;

  std::mutex _mutex;
  std::condition_variable _workersFinished;
  int32_t _busyWorkerCount;
};

std::ostream& operator<<(std::ostream&, const FeedForwardNetwork&);

class FeedForwardWorker
{
public:
  FeedForwardWorker(FeedForwardNetwork&);
  std::pair<uint32_t, double> EvaluateAccuracy(std::vector<Image*>::const_iterator begin, uint32_t count);
protected:
  void FeedForward(const Tensor& input);

  FeedForwardNetwork& _network;
  std::vector<Tensor> _activations;
};

class FeedForwardClassifier : public FeedForwardWorker
{
public:
  FeedForwardClassifier(FeedForwardNetwork& network)
	: FeedForwardWorker(network), _batchSize(0) {}
  FeedForwardClassifier(FeedForwardNetwork& network, std::vector<Image*>::const_iterator batchBegin,
	std::vector<uint32_t>::iterator resultsBegin, uint32_t batchSize)
	: FeedForwardWorker(network)
  {
	_batchBegin = batchBegin;
	_resultsBegin = resultsBegin;
	_batchSize = batchSize;
  }
  void ClassifyOnBackgroundThread();
  void Classify(std::vector<Image*>::const_iterator begin, std::vector<uint32_t>::iterator result, uint32_t count);
private:
  std::vector<Image*>::const_iterator _batchBegin;
  std::vector<uint32_t>::iterator _resultsBegin;
  uint32_t _batchSize;
};

class FeedForwardTrainer : public FeedForwardWorker
{
public:
  enum class Phases { Training, Testing, Finished };

  FeedForwardTrainer(FeedForwardNetwork&);

  void TrainOnBackgroundThread();
  void AddBatch(std::vector<Image*>::const_iterator begin, uint32_t batchSize);
  void SetActivity(Phases phase)
  {
	_currentPhase = phase;
	_numberCorrect = 0;
	_totalTrainingCost = 0.0;
	_totalTestingCost = 0.0;
  }
  void SignalTrainingDone()
  {
	std::unique_lock<std::mutex> lock(_mutex);
	_currentPhase = Phases::Finished;
	_nextBatchAvailable.notify_one();
  }

  void TrainOnMiniBatch(std::vector<Image*>::const_iterator begin, uint32_t batchSize);
  void BackPropagate(const Tensor& example, const Tensor& correctOutput);
  const std::vector<TensorPtr>& NablaB() const { return _nablaB; }
  const std::vector<TensorPtr>& NablaW() const { return _nablaW; }
  uint32_t NumberCorrect() const { return _numberCorrect; }
  double TotalTrainingCost() const { return _totalTrainingCost; }
  double TotalTestingCost() const { return _totalTestingCost; }
private:
  void WaitForNextBatch()
  {
	std::unique_lock<std::mutex> lock(_mutex);
	if (_batchSize == 0 && _currentPhase != Phases::Finished)
	  _nextBatchAvailable.wait(lock, [this] { return _batchSize > 0 || _currentPhase == Phases::Finished; });
  }
  std::vector<TensorPtr> _derivatives;
  std::vector<Tensor> _delta;
  std::vector<TensorPtr> _nablaB;
  std::vector<TensorPtr> _nablaW;
  std::vector<DropoutMaskPtr> _dropoutMasks;

  std::mutex _mutex;
  std::condition_variable _nextBatchAvailable;
  std::vector<Image*>::const_iterator _batchBegin;
  uint32_t _batchSize;
  uint32_t _numberCorrect;
  double _totalTrainingCost;
  double _totalTestingCost;
  std::atomic<Phases> _currentPhase;
};
