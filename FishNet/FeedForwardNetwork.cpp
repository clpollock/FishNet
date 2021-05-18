#include "stdafx.h"
#include "FeedForwardNetwork.h"
#include "CostFunction.h"
#include "ImageSet.h"
#include "ConvolutionalLayer.h"

static const char* magicString = "FishNet123";
static const uint16_t currentFileVersion = 6;

namespace
{

std::string FileNameBase(const std::string& saveDir, const std::string& networkName)
{
  auto startTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
#ifdef _WIN32
  struct tm time;
  localtime_s(&time, &startTime);
  auto localTime = &time;
#else
  auto localTime = std::localtime(&startTime);
#endif
  std::stringstream ss;
  ss << saveDir << networkName << '_'
	<< (1900 + localTime->tm_year)
	<< std::setfill('0') << std::setw(2) << (1 + localTime->tm_mon)
	<< std::setfill('0') << std::setw(2) << (localTime->tm_mday)
	<< '-'
	<< std::setfill('0') << std::setw(2) << localTime->tm_hour
	<< std::setfill('0') << std::setw(2) << localTime->tm_min
	<< std::setfill('0') << std::setw(2) << localTime->tm_sec;
  return ss.str();
}

}

FeedForwardNetwork::FeedForwardNetwork(const std::string& name, uint32_t inputChannelCount, uint32_t inputRows, uint32_t inputColumns,
  std::unique_ptr<::CostFunction> costFunction, uint32_t threadCount, uint16_t epochsTrained, double learningRate, double weightDecay)
  : _name(name), _costFunction(std::move(costFunction)), _inputChannelCount(inputChannelCount), _inputRows(inputRows),
	_inputColumns(inputColumns), _threadCount(threadCount), _epochsTrained(epochsTrained),
	_learningRate(learningRate), _weightDecay(weightDecay), _weightDecayMultiplier(1.0),
	_oneHotCategories(nullptr), _busyWorkerCount(0)
{
}

FeedForwardNetwork::~FeedForwardNetwork()
{
}

void FeedForwardNetwork::AddFullyConnectedLayer(uint32_t layerSize, std::unique_ptr<::ActivationFunction>&& activationFunction,
  double keepProbability)
{
  if (keepProbability > 1.0 || keepProbability <= 0.0)
	throw std::runtime_error("Keep probability must be greater than 0 and no greater than one.");
  uint32_t inputSize;
  if (_layers.empty())
  {
	inputSize = _inputChannelCount * _inputRows * _inputColumns;
  }
  else
  {
	const Layer& prevLayer = *_layers.back();
	inputSize = prevLayer.OutputPlanes() * prevLayer.OutputRows() * prevLayer.OutputColumns();
  }

  double prevLayerKeepProbability = 1.0;
  if (!_layers.empty())
  {
	const auto prevLayer = dynamic_cast<FullyConnectedLayer*>(_layers.back().get());
	if (prevLayer)
	  prevLayerKeepProbability = prevLayer->KeepProbability();
  }
  _layers.emplace_back(std::make_unique<FullyConnectedLayer>(inputSize, layerSize, std::move(activationFunction), keepProbability,
	prevLayerKeepProbability));
}

void FeedForwardNetwork::AddConvolutionalLayer(uint32_t filterCount, uint32_t filterSize, uint32_t stride, uint32_t zeroPadding,
  std::unique_ptr<::ActivationFunction>&& activationFunction)
{
  if (zeroPadding >= filterSize)
	throw std::runtime_error("Zero padding must be less than the size of the filter.");
  uint32_t inputChannelCount;
  uint32_t inputColumns;
  uint32_t inputRows;
  if (_layers.empty())
  {
	inputChannelCount = _inputChannelCount;
	inputRows = _inputRows;
	inputColumns = _inputColumns;
  }
  else
  {
	const Layer& prevLayer = *_layers.back();
	if (dynamic_cast<const FullyConnectedLayer*>(&prevLayer))
	  throw std::runtime_error("A convolutional layer cannot follow a fully connected layer.");
	inputChannelCount = prevLayer.OutputPlanes();
	inputRows = prevLayer.OutputRows();
	inputColumns = prevLayer.OutputColumns();
  }
  _layers.emplace_back(std::make_unique<ConvolutionalLayer>(inputChannelCount, inputRows, inputColumns, filterCount, filterSize,
	stride, zeroPadding, std::move(activationFunction)));
}

void FeedForwardNetwork::AddMaxPoolingLayer()
{
  if (_layers.empty())
	throw std::runtime_error("A max pooling layer cannot be the first layer in the network.");

  uint32_t inputChannelCount;
  uint32_t inputColumns;
  uint32_t inputRows;
  if (_layers.empty())
  {
	inputChannelCount = _inputChannelCount;
	inputRows = _inputRows;
	inputColumns = _inputColumns;
  }
  else
  {
	const Layer& prevLayer = *_layers.back();
	if (dynamic_cast<const FullyConnectedLayer*>(&prevLayer))
	  throw std::runtime_error("A max pooling layer cannot follow a fully connected layer.");
	inputChannelCount = prevLayer.OutputPlanes();
	inputRows = prevLayer.OutputRows();
	inputColumns = prevLayer.OutputColumns();
  }
  _layers.emplace_back(std::make_unique<MaxPoolingLayer>(inputChannelCount, inputRows, inputColumns));
}

void FeedForwardNetwork::Save(const std::string& fileName)
{
  std::ofstream os(fileName.c_str(), std::ofstream::trunc|std::ofstream::binary);
  if (!os.good())
  {
	LOG(Error) << "Failed to open file " << fileName << " for writing.";
	return;
  }
  os.write(magicString, 10);
  os.write((const char*)&currentFileVersion, sizeof(uint16_t));
  uint16_t nameLength = static_cast<uint16_t>(_name.size());
  os.write((const char*)&nameLength, sizeof(uint16_t));
  os.write(_name.c_str(), _name.size());
  os.write((const char*)&_inputChannelCount, sizeof(uint32_t));
  os.write((const char*)&_inputRows, sizeof(uint32_t));
  os.write((const char*)&_inputColumns, sizeof(uint32_t));
  os.put((char)_costFunction->Type());
  os.write((const char*)&_epochsTrained, sizeof(uint32_t));
  os.write((const char*)&_learningRate, sizeof(double));
  os.write((const char*)&_weightDecay, sizeof(double));
  uint16_t numberOfLayers = static_cast<uint16_t>(_layers.size());
  os.write((const char*)&numberOfLayers, sizeof(uint16_t));
  for (const auto& layer : _layers)
	layer->Save(os);
  LOG(Info) << "Saved to " << fileName;
}

std::unique_ptr<FeedForwardNetwork> FeedForwardNetwork::Load(const std::string& fileName, uint32_t threadCount)
{
  std::ifstream is(fileName.c_str(), std::ofstream::binary);
  if (!is.good())
	throw std::runtime_error("Failed to open file " + fileName + " for reading.");
  char magic[10];
  is.read(magic, 10);
  if (strncmp(magicString, magic, 10) != 0)
	throw std::runtime_error(fileName + " does not appear to be a FishNet file.");
  uint16_t versionNumber = 1;
  is.read((char*)&versionNumber, sizeof(uint16_t));
  if (versionNumber > currentFileVersion)
  {
	std::stringstream msg;
	msg << fileName << " was saved by a newer version  of FishNet. Please upgrade your version to load it.";
	throw std::runtime_error(msg.str());
  }
  std::string name;
  if (versionNumber >= 6)
  {
    uint16_t nameLength = 0;
	is.read((char*)&nameLength, sizeof(uint16_t));
	auto s = std::make_unique<char[]>(nameLength);
	is.read(s.get(), nameLength);
	name = s.get();
  }
  uint32_t inputChannelCount, inputRows, inputColumns;
  is.read((char*)&inputChannelCount, sizeof(uint32_t));
  is.read((char*)&inputRows, sizeof(uint32_t));
  is.read((char*)&inputColumns, sizeof(uint32_t));
  std::unique_ptr<::CostFunction> costFunction;
  switch (static_cast<::CostFunction::Types>(is.get()))
  {
	case ::CostFunction::Types::CrossEntropy:
	  costFunction.reset(new CrossEntropyCostFunction);
	  break;
	default:
	  throw std::runtime_error(fileName + " contains an unrecognized cost function.");
  }
  // Number of epochs for which the network has been trained.
  uint32_t epochsTrained = 0;
  if (versionNumber < 5)
  {
	uint16_t tmp = 0;
	is.read((char*)&tmp, sizeof(uint16_t));
	epochsTrained = tmp;
  }
  else
  {
	is.read((char*)&epochsTrained, sizeof(uint32_t));
  }
  double learningRate = 0.05;
  is.read((char*)&learningRate, sizeof(double));
  double weightDecay = 0.0;
  if (versionNumber >= 4)
	is.read((char*)&weightDecay, sizeof(double));
  auto network = std::make_unique<FeedForwardNetwork>(name, inputChannelCount, inputRows, inputColumns,
	std::move(costFunction), threadCount, epochsTrained, learningRate, weightDecay);
  // Load network layers.
  uint16_t numberOfLayers;
  is.read((char*)&numberOfLayers, sizeof(uint16_t));
  double prevLayerKeepProbability = 1.0;
  for (uint16_t li = 0; li < numberOfLayers; ++li)
  {
	auto layer = Layer::Load(is, inputChannelCount, inputRows, inputColumns, prevLayerKeepProbability);
	inputChannelCount = layer->OutputPlanes();
	inputRows = layer->OutputRows();
	inputColumns = layer->OutputColumns();
	prevLayerKeepProbability = layer->KeepProbability();
	network->AddLayer(std::move(layer));
  }
  LOG(Info) << "Loaded network " << fileName;
  return network;
}

std::vector<uint32_t> FeedForwardNetwork::Classify(const ImageSet& imageSet)
{
  // This should never happen unless the user really doesn't know what he's doing.
  if (imageSet.TestSet().size() < _threadCount)
	throw std::runtime_error("Number of threads cannot be greater than the test set size.");
  for (auto& layer : _layers)
	layer->SwitchToTestingWeights();
  // Create _threadCount - 1 testers to run on background threads because we also test on the foreground thread.
  std::vector<FeedForwardClassifier> backgroundTesters;
  backgroundTesters.reserve(_threadCount - 1);
  _backgroundThreads.reserve(_threadCount - 1);
  _backgroundThreads.clear();
  
  std::vector<Image*>::const_iterator batchBegin = imageSet.TestSet().cbegin();
  uint32_t testSetSize = static_cast<uint32_t>(imageSet.TestSet().size());
  uint32_t perThreadSize = testSetSize / _threadCount;
  uint32_t remainder = testSetSize % _threadCount;
  std::vector<uint32_t> results(testSetSize, 0);
  auto batchResults = results.begin();
  for (uint32_t t = 0; t < _threadCount - 1; ++t)
  {
	uint32_t thisBatchSize = perThreadSize;
	if (remainder > 0)
	{
	  ++thisBatchSize;
	  --remainder;
	}
	backgroundTesters.emplace_back(*this, batchBegin, batchResults, thisBatchSize);
	_backgroundThreads.emplace_back(&FeedForwardClassifier::ClassifyOnBackgroundThread, &backgroundTesters.back());
	batchBegin += thisBatchSize;
	batchResults += thisBatchSize;
  }

  FeedForwardClassifier foregroundTester(*this);
  foregroundTester.Classify(batchBegin, batchResults, perThreadSize);

  for (auto& thread : _backgroundThreads)
	thread.join();
  _backgroundThreads.clear();
  
  return results;
}

void FeedForwardNetwork::SaveAccuracyStatistics(const ImageSet& imageSet, std::ostream& os)
{
  std::vector<uint32_t> classifications = Classify(imageSet);

  std::vector<std::vector<uint32_t>> classificationsByType;
  for (uint32_t i = 0; i < imageSet.Categories().size(); ++i)
	classificationsByType.emplace_back(imageSet.Categories().size(), 0);

  uint32_t numberCorrect = 0;
  for (size_t ii = 0; ii < imageSet.TestSet().size(); ++ii)
  {
	const Image& image = *imageSet.TestSet()[ii];
	uint32_t selectedCategory = classifications[ii];
	++classificationsByType[image.Category()][selectedCategory];
	if (selectedCategory == image.Category())
	  ++numberCorrect;
  }

  os << "Actual Category";
  for (const auto& category : imageSet.Categories())
	os << ",Predicted " << category;
  os << std::endl;
  for (uint32_t i = 0; i < imageSet.Categories().size(); ++i)
  {
	os << imageSet.Categories()[i];
	const std::vector<uint32_t>& classifications = classificationsByType[i];
	for (uint32_t j = 0; j < imageSet.Categories().size(); ++j)
	  os << ',' << classifications[j];
	os << std::endl;
  }

  os << std::endl << "Overall accuracy," << numberCorrect << std::endl;
}

void FeedForwardNetwork::SaveWeightStatistics(std::ostream& os) const
{
  os << "Weight and Bias Statistics" << std::endl
	<< "Layer,Maximum Weight,Minimum Weight,Average Weight,Maximum Bias,Minimum Bias,Average Bias" << std::endl;
  for (int i = 0; i < _layers.size(); ++i)
  {
	const WeightedLayer* wl = dynamic_cast<const WeightedLayer*>(_layers[i].get());
	if (wl)
	{
	  double max, min, avg;
	  wl->Weights().GetStatistics(max, min, avg);
	  os << i << ',' << max << ',' << min << ',' << avg << ',';
	  wl->Biases().GetStatistics(max, min, avg);
	  os << max << ',' << min << ',' << avg << std::endl;
	}
  }
}

void FeedForwardNetwork::SaveArchitecture(std::ostream& os) const
{
  os << "Network" << std::endl
	<< "Layer,Layer Size,Filter Count,Filter Size,Stride,Padding,Dropout,Activation,Leakiness" << std::endl;
  for (const auto& layer : _layers)
	layer->SaveArchitecture(os);
}

void FeedForwardNetwork::Train(const ImageSet& imageSet, uint32_t epochs, uint32_t giveUpAfter, uint32_t miniBatchSize,
  double learningRateDecay, double learningRateDecayPoint, const std::string& saveDir)
{
  if (imageSet.TrainingSet().size() < _threadCount)
	throw std::runtime_error("Number of threads cannot be greater than the training set size.");
  if (imageSet.TestSet().size() < _threadCount)
	throw std::runtime_error("Number of threads cannot be greater than the test set size.");
  LOG(Info) << "Training on " << imageSet.Name() << " for " << epochs << " epochs.";
  if (_epochsTrained > 0)
  {
	epochs += _epochsTrained;
	LOG(Info) << "Network has already been trained for " << _epochsTrained << " epochs.";
  }
  if (giveUpAfter < epochs)
	LOG(Info) << "Will stop training after " << giveUpAfter << " epochs without any improvement in accuracy.";
  LOG(Info) << "Using " << _threadCount << " threads.";
  LOG(Info) << "Learning rate: " << _learningRate << ", learning rate decay: " << learningRateDecay	<< ", weight decay: " << _weightDecay;
  LOG(Info) << "Network architecture:" << std::endl << *this;

  std::string fileNameBase = FileNameBase(saveDir, _name);
  // Save the learning statistics as we go.
  std::string statsFileName = fileNameBase + ".csv";
  std::ofstream statsFile(statsFileName);
  if (!statsFile.good())
	throw std::runtime_error("Failed to open file " + statsFileName + " for writing.");
  statsFile << "Dataset," << imageSet.Name() << std::endl
	<< "Learning rate," << _learningRate << std::endl;
  if (learningRateDecay != 0.0)
  {
	statsFile << "Learning rate decay," << learningRateDecay << std::endl
	  << "Learning rate decay point," << learningRateDecayPoint << std::endl;
  }
  if (_weightDecay != 0.0)
  {
	_weightDecayMultiplier = 1.0 - (_weightDecay *  _learningRate);
	statsFile << "Weight decay," << _weightDecay << std::endl;
  }
  statsFile << "Minibatch size," << miniBatchSize << std::endl << std::endl;
  SaveArchitecture(statsFile);
  statsFile << std::endl << "Epoch,Training Loss,Testing Loss,Accuracy" << std::endl;

  // Create and initialize the weights of each layer. This won't do anything if the weights have been loaded from a file.
  for (auto& layer : _layers)
	layer->InitializeWeights();
  _oneHotCategories = &imageSet.OneHotCategories();

  std::vector<Image*> trainingData(imageSet.TrainingSet().begin(), imageSet.TrainingSet().end());
  static std::random_device rd;
  static std::mt19937 shuffler(rd());

  // Set previousTrainingCost very high so that learning rate decay won't be triggered after the first epoch.
  double previousTrainingCost = 1e6;
  uint32_t highestNumberCorrect = 0;
  uint32_t bestEpoch = _epochsTrained;

  StartTrainers();

  while (_epochsTrained < epochs)
  {
	auto trainingStart = std::chrono::steady_clock::now();
	// Randomly shuffle the training data.
    std::shuffle(trainingData.begin(), trainingData.end(), shuffler);
	double trainingCost = TrainForOneEpoch(trainingData, miniBatchSize);
	auto trainingEnd = std::chrono::steady_clock::now();
	LOG(Info) << "Training epoch " << _epochsTrained << " completed in "
	  << std::chrono::duration_cast<std::chrono::milliseconds>(trainingEnd - trainingStart).count()
	  << " ms. Average training cost: " << trainingCost << std::endl;
	// When we're training with dropout, we need to switch to the weights without dropout for testing.
	for (auto& layer : _layers)
	  layer->SwitchToTestingWeights();
	auto [numberCorrect, averageTestingCost] = TestDuringTraining(imageSet);
	for (auto& layer : _layers)
	  layer->SwitchToTrainingWeights();
	// Log test results.
	auto testingEnd = std::chrono::steady_clock::now();
	LOG(Info) << "Testing completed in "
	  << std::chrono::duration_cast<std::chrono::milliseconds>(testingEnd - trainingEnd).count() << " ms." << std::endl
	  << "Correctly determined " << numberCorrect << " out of " << imageSet.TestSet().size()
	  << ", average testing cost: " << averageTestingCost << std::endl;
	// Save learning statistics to the CSV file.
	statsFile << _epochsTrained << ',' << trainingCost << ',' << averageTestingCost << ',' << numberCorrect << std::endl << std::flush;

	if (numberCorrect > highestNumberCorrect)
	{
	  highestNumberCorrect = numberCorrect;
	  bestEpoch = _epochsTrained;
	  Save(fileNameBase + '_' + std::to_string(_epochsTrained) + ".fish");
	}
	else if (_epochsTrained - bestEpoch >= giveUpAfter)
	{
	  LOG(Info) << "Stopping training after " << giveUpAfter << " epochs with no improvement in accuracy." << std::endl;
	  break;
	}

	if (learningRateDecay != 0.0 && trainingCost / previousTrainingCost > learningRateDecayPoint)
	{
	  // If the training cost didn't improve after the last epoch, reduce the learning rate.
	  _learningRate *= (1.0 - learningRateDecay);
	  if (_weightDecay != 0.0)
		_weightDecayMultiplier = 1.0 - (_weightDecay *  _learningRate);
	  LOG(Info) << "Reduced learning rate to " << _learningRate;
	}
	previousTrainingCost = trainingCost;
  }

  StopTrainers();

  statsFile << std::endl;
  SaveWeightStatistics(statsFile);
  statsFile << std::endl << "Network Classifications" << std::endl;
  SaveAccuracyStatistics(imageSet, statsFile);
}

void FeedForwardNetwork::SignalWorkerFinished()
{
  std::unique_lock<std::mutex> lock(_mutex);
  if (--_busyWorkerCount <= 0)
  {
	if (_busyWorkerCount < 0)
	  throw std::runtime_error("Busy worker count has become negative!");
	_workersFinished.notify_one();
  }
}

double FeedForwardNetwork::TrainForOneEpoch(const std::vector<Image*>& trainingData, uint32_t miniBatchSize)
{
  auto previousReportTime = std::chrono::steady_clock::now();

  _foregroundTrainer->SetActivity(FeedForwardTrainer::Phases::Training);
  for (auto& trainer : _backgroundTrainers)
	trainer->SetActivity(FeedForwardTrainer::Phases::Training);

  std::vector<Image*>::const_iterator begin = trainingData.cbegin();

  uint32_t remaining = static_cast<uint32_t>(trainingData.size());
  while (remaining > 0)
  {
	if (remaining < miniBatchSize)
	  miniBatchSize = remaining;

	// If there are fewer remaining examples than background threads, some threads will be unused this time.
	uint32_t backgroundWorkerCount = std::min(miniBatchSize, static_cast<uint32_t>(_backgroundThreads.size()));
	_busyWorkerCount = backgroundWorkerCount;

	uint32_t perThreadSize = miniBatchSize / _threadCount;
	uint32_t remainder = miniBatchSize % _threadCount;

	for (auto& trainer : _backgroundTrainers)
	{
	  uint32_t thisBatchSize = perThreadSize;
	  if (remainder > 0)
	  {
		++thisBatchSize;
		--remainder;
	  }
	  else if (thisBatchSize == 0)
		break;
	  trainer->AddBatch(begin, thisBatchSize);
	  begin += thisBatchSize;
	}

	if (perThreadSize > 0)
	{
	  _foregroundTrainer->TrainOnMiniBatch(begin, perThreadSize);
	  begin += perThreadSize;
	}

	WaitForBackgroundTrainers();

	double scalar = _learningRate / miniBatchSize;
	size_t li = 0;
	for (auto& layer : _layers)
	{
	  auto wl = dynamic_cast<WeightedLayer*>(layer.get());
	  if (wl)
	  {
		if (_weightDecayMultiplier != 1.0)
		  wl->DecayWeights(_weightDecayMultiplier);

		for (size_t ti = 0; ti < backgroundWorkerCount; ++ti)
		{
		  const FeedForwardTrainer& trainer = *_backgroundTrainers[ti];
		  wl->UpdateWeightsAndBiases(*trainer.NablaW()[li], *trainer.NablaB()[li], scalar);
		}
		if (perThreadSize > 0)
		  wl->UpdateWeightsAndBiases(*_foregroundTrainer->NablaW()[li], *_foregroundTrainer->NablaB()[li], scalar);
	  }
	  ++li;
	}

	remaining -= miniBatchSize;
	if (remaining > 0)
	{
	  // Report progress every two minutes.
	  auto now = std::chrono::steady_clock::now();
	  if (std::chrono::duration_cast<std::chrono::seconds>(now - previousReportTime).count() >= 120)
	  {
		LOG(Info) << remaining << " training examples remaining in epoch." << std::endl;
		previousReportTime = now;
	  }
	}
  }

  ++_epochsTrained;
  double trainingCost = _foregroundTrainer->TotalTrainingCost();
  for (const auto& trainer : _backgroundTrainers)
	trainingCost += trainer->TotalTrainingCost();
  trainingCost /= static_cast<double>(trainingData.size());
  return trainingCost;
}

std::pair<uint32_t, double> FeedForwardNetwork::TestDuringTraining(const ImageSet& imageSet)
{
  std::vector<Image*>::const_iterator begin = imageSet.TestSet().cbegin();
  uint32_t testSetSize = static_cast<uint32_t>(imageSet.TestSet().size());
  uint32_t perThreadSize = testSetSize / _threadCount;
  uint32_t remainder = testSetSize % _threadCount;

  _busyWorkerCount = std::min(testSetSize, _threadCount - 1);

  _foregroundTrainer->SetActivity(FeedForwardTrainer::Phases::Testing);
  for (auto& trainer : _backgroundTrainers)
  {
	uint32_t thisBatchSize = perThreadSize;
	if (remainder > 0)
	{
	  ++thisBatchSize;
	  --remainder;
	}
	else if (thisBatchSize == 0)
	  break;
	trainer->SetActivity(FeedForwardTrainer::Phases::Testing);
	trainer->AddBatch(begin, thisBatchSize);
	begin += thisBatchSize;
  }

  std::pair<uint32_t, double> result = _foregroundTrainer->EvaluateAccuracy(begin, perThreadSize);
  WaitForBackgroundTrainers();

  for (const auto& trainer : _backgroundTrainers)
  {
	result.first += trainer->NumberCorrect();
	result.second += trainer->TotalTestingCost();
  }

  result.second /= static_cast<double>(testSetSize);
  return result;
}

void FeedForwardNetwork::StartTrainers()
{
  _foregroundTrainer = std::make_unique<FeedForwardTrainer>(*this);
  // Create _threadCount - 1 trainers to run on background threads because we also train on the foreground thread.
  _backgroundTrainers.reserve(_threadCount - 1);
  _backgroundThreads.reserve(_threadCount - 1);
  for (uint32_t t = 0; t < _threadCount - 1; ++t)
  {
	_backgroundTrainers.emplace_back(std::make_unique<FeedForwardTrainer>(*this));
	_backgroundThreads.emplace_back(&FeedForwardTrainer::TrainOnBackgroundThread, _backgroundTrainers.back().get());
  }
}

void FeedForwardNetwork::StopTrainers()
{
  _foregroundTrainer = nullptr;
  for (auto& trainer : _backgroundTrainers)
	trainer->SignalTrainingDone();
  for (auto& thread : _backgroundThreads)
	thread.join();
  _backgroundTrainers.clear();
  _backgroundThreads.clear();
}

std::ostream& operator<<(std::ostream& os, const FeedForwardNetwork& network)
{
  for (uint32_t layer = 0; layer < network.Layers().size(); ++layer)
  {
	os << "\tLayer " << layer << ": ";
	network.Layers()[layer]->Description(os);
	os << std::endl;
  }
  return os;
}

FeedForwardWorker::FeedForwardWorker(FeedForwardNetwork& network)
  : _network(network)
{
  for (const auto& layer : _network.Layers())
	_activations.emplace_back(layer->OutputPlanes(), layer->OutputRows(), layer->OutputColumns());
}

std::pair<uint32_t, double> FeedForwardWorker::EvaluateAccuracy(std::vector<Image*>::const_iterator begin, uint32_t count)
{
  // Count the number of test examples for which the network predicted the correct class.
  const Tensor& outputs = _activations.back();
  uint32_t numberCorrect = 0;
  double totalCost = 0.0;
  auto end = begin + count;
  while (begin != end)
  {
	const Image& example = **begin;
	FeedForward(example.Inputs());
	// The network's predicted class is the one that produced the highest activation in the output layer.
	if (outputs.HighestValueIndex() == example.Category())
	  ++numberCorrect;
	// Also calculate total cost (error) for this example.
	totalCost += _network.CostFunction().TotalCost(outputs, (*_network.OneHotCategories())[example.Category()]);
	++begin;
  }
  return std::make_pair(numberCorrect, totalCost);
}

void FeedForwardWorker::FeedForward(const Tensor& input)
{
  const Tensor* layerInput = &input;
  auto layerActivations = _activations.begin();
  for (const auto& layer : _network.Layers())
  {
	layer->FeedForward(*layerInput, *layerActivations, nullptr);
	auto wl = dynamic_cast<WeightedLayer*>(layer.get());
	if (wl)
	  wl->ApplyActivationFunction(*layerActivations);
	layerInput = &*layerActivations;
	++layerActivations;
  }
}

void FeedForwardClassifier::ClassifyOnBackgroundThread()
{
  Classify(_batchBegin, _resultsBegin, _batchSize);
}

void FeedForwardClassifier::Classify(std::vector<Image*>::const_iterator begin, std::vector<uint32_t>::iterator result, uint32_t count)
{
  const Tensor& outputs = _activations.back();
  auto end = begin + count;
  while (begin != end)
  {
	const Image& example = **begin;
	FeedForward(example.Inputs());
	*result = outputs.HighestValueIndex();
	++begin;
	++result;
  }
}

FeedForwardTrainer::FeedForwardTrainer(FeedForwardNetwork& network)
  : FeedForwardWorker(network),
	_batchSize(0), _numberCorrect(0), _totalTrainingCost(0.0), _totalTestingCost(0.0),
	_currentPhase(Phases::Training)
{
  for (const auto& layer : _network.Layers())
  {
	_delta.emplace_back(layer->OutputPlanes(), layer->OutputRows(), layer->OutputColumns());
	auto wl = dynamic_cast<WeightedLayer*>(layer.get());
	if (wl)
	{
	  _derivatives.emplace_back(std::make_unique<Tensor>(layer->OutputPlanes(), layer->OutputRows(), layer->OutputColumns()));
	  _nablaB.emplace_back(std::make_unique<Tensor>(wl->Biases().Size()));
	  const Tensor& weights = wl->Weights();
	  _nablaW.emplace_back(std::make_unique<Tensor>(weights.Hyperplanes(), weights.Planes(), weights.Rows(), weights.Columns()));
	}
	else
	{
	  _derivatives.emplace_back(nullptr);
	  _nablaB.emplace_back(nullptr);
	  _nablaW.emplace_back(nullptr);
	}
	// Create a DropoutMask for all layers that use dropout.
	auto fcn = dynamic_cast<FullyConnectedLayer*>(layer.get());
	if (fcn && fcn->KeepProbability() < 1.0)
	  _dropoutMasks.emplace_back(std::make_unique<DropoutMask>(fcn->KeepProbability(), fcn->Weights().Rows()));
	else
	  _dropoutMasks.emplace_back(nullptr);
  }
}

void FeedForwardTrainer::TrainOnBackgroundThread()
{
  do
  {
	WaitForNextBatch();
	while (_currentPhase == Phases::Training)
	{
	  TrainOnMiniBatch(_batchBegin, _batchSize);
	  _batchSize = 0;
	  _network.SignalWorkerFinished();
	  WaitForNextBatch();
	}
	while (_currentPhase == Phases::Testing)
	{
	  std::pair<uint32_t, double> result = EvaluateAccuracy(_batchBegin, _batchSize);
	  _numberCorrect += result.first;
	  _totalTestingCost += result.second;
	  _batchSize = 0;
	  _network.SignalWorkerFinished();
	  WaitForNextBatch();
	}
  } while (_currentPhase != Phases::Finished);
}

void FeedForwardTrainer::AddBatch(std::vector<Image*>::const_iterator batchBegin, uint32_t batchSize)
{
  std::unique_lock<std::mutex> lock(_mutex);
  _batchBegin = batchBegin;
  _batchSize = batchSize;
  _nextBatchAvailable.notify_one();
}

void FeedForwardTrainer::TrainOnMiniBatch(std::vector<Image*>::const_iterator begin, uint32_t batchSize)
{
  for (auto& t : _nablaB)
  {
	if (t)
	  t->SetAllToZero();
  }
  for (auto& t : _nablaW)
  {
	if (t)
	  t->SetAllToZero();
  }

  auto end = begin + batchSize;
  while (begin != end)
  {
	const Image& example = **begin;
	BackPropagate(example.Inputs(), (*_network.OneHotCategories())[example.Category()]);
	++begin;
  }
}

void FeedForwardTrainer::BackPropagate(const Tensor& example, const Tensor& correctOutput)
{
  // Feed the example through the network so that we can
  // calculate the cost at the output layer.
  const Tensor* layerInput = &example;
  auto layerActivations = _activations.begin();
  auto layerDerivatives = _derivatives.begin();
  auto layerDropoutMask = _dropoutMasks.begin();
  for (const auto& layer : _network.Layers())
  {
	auto dropoutMask = layerDropoutMask->get();
	if (dropoutMask)
	  dropoutMask->Randomize();
	layer->FeedForward(*layerInput, *layerActivations, dropoutMask);
	auto wl = dynamic_cast<WeightedLayer*>(layer.get());
	if (wl)
	{
	  if (wl->ActivationFunction())
	  {
		wl->ActivationFunction()->ApplyDerivative(*layerActivations, **layerDerivatives);
		wl->ActivationFunction()->Apply(*layerActivations);
	  }
	  else
	  {
		**layerDerivatives = *layerActivations;
	  }
	}
	layerInput = &*layerActivations;
	++layerActivations;
	++layerDerivatives;
	++layerDropoutMask;
  }

  _totalTrainingCost += _network.CostFunction().TotalCost(_activations.back(), correctOutput);
  // Now do the backpropagation.
  // Calculate the error in the output layer.
  _network.CostFunction().Derivatives(_activations.back(), correctOutput, _delta.back());

  for (size_t li = _network.Layers().size() - 1; li > 0; --li)
  {
	Layer* layer = _network.Layers()[li].get();
	auto wl = dynamic_cast<WeightedLayer*>(layer);
	if (wl)
	{
	  _delta[li].ComponentWiseMultiply(*_derivatives[li]);
	  auto dropoutMask = _dropoutMasks[li].get();
	  wl->BackpropagateError(_delta[li], _delta[li - 1], dropoutMask);
	  wl->UpdateWeightAndBiasErrors(_delta[li], _activations[li - 1], *_nablaW[li], *_nablaB[li], dropoutMask);
	}
	else
	{
	  auto mpl = dynamic_cast<MaxPoolingLayer*>(layer);
	  if (mpl)
		mpl->BackpropagateError(_activations[li], _activations[li - 1], _delta[li], _delta[li - 1]);
	}
  }
  // First layer must always be a WeightedLayer.
  _delta.front().ComponentWiseMultiply(*_derivatives.front());
  static_cast<WeightedLayer&>(*_network.Layers().front()).UpdateWeightAndBiasErrors(_delta.front(),
	example, *_nablaW.front(), *_nablaB.front(), _dropoutMasks.front().get());
}
