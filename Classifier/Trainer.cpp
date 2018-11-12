#include "stdafx.h"
#include "Trainer.h"

class NetworkParamError : public std::exception
{
  public:
	NetworkParamError(const std::string& error, int lineNo)
	  : _error(error), _lineNo(lineNo) {}
	const std::string& Error() const { return _error; }
	int LineNo() const { return _lineNo; }
  private:
	std::string _error;
	int _lineNo;
};

std::ostream& operator<<(std::ostream& os, const Job& job)
{
  os << "Train for " << job.Epochs() << " epochs on " << job.DataSet().Name() << " dataset." << std::endl;
  if (job.GiveUpAfter() < job.Epochs())
	os << "Stop training after " << job.GiveUpAfter() << " epochs without progress." << std::endl;
  os << "Mini batch size: " << job.MiniBatchSize() << ", learning rate: " << job.Network().LearningRate() << std::endl;
  if (job.LearningRateDecay() != 0.0)
  {
	os << "Learning rate decay: " << job.LearningRateDecay()
	  << ", learning rate decay point: " << job.LearningRateDecayPoint() << std::endl;
  }
  if (job.Network().WeightDecay() != 0.0)
	os << "Weight  decay: " << job.Network().WeightDecay() << std::endl;
  if (!job.Network().Name().empty())
	os << "Network name: " << job.Network().Name() << std::endl;
  os << "Network architecture:" << std::endl << job.Network();
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<std::unique_ptr<Job>>& jobs)
{
  for (const auto& job : jobs)
	os << *job << std::endl;
  return os;
}

static std::string StripDir(const std::string& path)
{
  size_t i = path.find_last_of("\\/");
  if (i != std::string::npos)
	return path.substr(i + 1);
  return path;
}

void Trainer::LoadJobList(const std::string& fileName)
{
  std::ifstream is(fileName, std::ifstream::in);
  if (!is)
	throw std::runtime_error("Unable to read file " + fileName);
  LOG(Info) << "Loading jobs from " << fileName;
  uint32_t epochs = 0;
  uint32_t giveUpAfter = std::numeric_limits<uint32_t>::max();
  uint32_t miniBatchSize = 0;
  double learningRate = 0.0;
  double learningRateDecay = 0.0;
  double learningRateDecayPoint = 0.0;
  double weightDecay = 0.0;

  const ImageSet* imageSet = nullptr;

  std::string line;
  std::vector<std::string> fields;
  int lineNo = 0;
  while (!is.eof())
  {
	try
	{
	  std::getline(is, line);
	  ++lineNo;
	  if (line.empty() || line.front() == '#')
		continue;
	  StringUtils::SplitCSV(fields, line);
	  if (fields.front().empty())
		continue;
	  std::string& first = fields.front();
	  StringUtils::ToLower(first);
	  if (first == "dataset")
	  {
		if (fields.size() < 2 || fields[1].empty())
		  throw std::runtime_error("Dataset name is missing.");
		StringUtils::ToLower(fields[1]);
		imageSet = &_imageSetLoader.Load(fields[1]);
	  }
	  else if (first == "epochs")
	  {
		if (fields.size() < 2 || fields[1].empty())
		  throw std::runtime_error("Number of epochs is missing.");
		epochs = std::stoi(fields[1]);
		if (epochs < 1)
		  throw std::runtime_error("Number of epochs must be at least 1.");
	  }
	  else if (first == "give up after")
	  {
		if (fields.size() < 2 || fields[1].empty())
		  throw std::runtime_error("You must specify how many epochs without progress to give up after.");
		giveUpAfter = std::stoi(fields[1]);
	  }
	  else if (first == "learning rate")
	  {
		if (fields.size() < 2 || fields[1].empty())
		  throw std::runtime_error("Learning rate is missing.");
		learningRate = std::stod(fields[1]);
		if (learningRate <= 0.0)
		  throw std::runtime_error("Learning rate must be greater than 0.");
	  }
	  else if (first == "learning rate decay")
	  {
		if (fields.size() < 2 || fields[1].empty())
		  throw std::runtime_error("Learning rate decay is missing.");
		learningRateDecay = std::stod(fields[1]);
		if (learningRateDecay >= 1.0 || learningRateDecay < 0.0)
		  throw std::runtime_error("Learning rate decay must be greater than or equal to 0 and less than 1.");
	  }
	  else if (first == "learning rate decay point")
	  {
		if (fields.size() < 2 || fields[1].empty())
		  throw std::runtime_error("Learning rate decay point is missing.");
		learningRateDecayPoint = std::stod(fields[1]);
		if (learningRateDecayPoint < 0.0)
		  throw std::runtime_error("Learning rate decay point must be greater than or equal to 0.");
	  }
	  else if (first == "minibatch size" || first == "minibatch")
	  {
		if (fields.size() < 2 || fields[1].empty())
		  throw std::runtime_error("Minibatch size is missing.");
		miniBatchSize = std::stoi(fields[1]);
		if (miniBatchSize < 1)
		  throw std::runtime_error("Minibatch size must be at least 1.");
	  }
	  else if (first == "network")
	  {
		if (!imageSet)
		  throw std::runtime_error("No dataset has been specified.");
		if (miniBatchSize == 0)
		  throw std::runtime_error("Minibatch size has not been specified.");
		if (learningRate == 0.0)
		  throw std::runtime_error("Learning rate has not been specified.");
		if (epochs == 0)
		  throw std::runtime_error("Epochs has not been specified.");
		std::string name = fields.size() >= 2 && !fields[1].empty() ? fields[1] : imageSet->Name();
		_jobs.emplace_back(std::make_unique<Job>(*imageSet, LoadNetwork(name, is, lineNo, *imageSet, learningRate, weightDecay),
		  epochs, giveUpAfter, miniBatchSize, learningRateDecay, learningRateDecayPoint));
	  }
	  else if (first == "network file")
	  {
		if (fields.size() < 2 || fields[1].empty())
		  throw std::runtime_error("Network file name is missing.");
		if (!imageSet)
		  throw std::runtime_error("No dataset has been specified.");
		if (miniBatchSize == 0)
		  throw std::runtime_error("Minibatch size has not been specified.");
		auto network = FeedForwardNetwork::Load(fields[1], _threadCount);
		if (learningRate != 0.0)
		  network->LearningRate(learningRate);
		if (epochs == 0)
		  throw std::runtime_error("Epochs has not been specified.");
		if (weightDecay != 0.0)
		  network->WeightDecay(weightDecay);
		if (network->Name().empty())
		  network->Name(imageSet->Name());
		_jobs.emplace_back(std::make_unique<Job>(*imageSet, std::move(network), epochs,
		  giveUpAfter, miniBatchSize, learningRateDecay, learningRateDecayPoint));
	  }
	  else if (first == "weight decay")
	  {
		if (fields.size() < 2 || fields[1].empty())
		  throw std::runtime_error("Weight decay is missing.");
		weightDecay = std::stod(fields[1]);
		if (weightDecay > 1.0 || weightDecay <= 0.0)
		  throw std::runtime_error("Weight decay must be greater than or equal to 0 and less than 1.");
	  }
	  else if (!first.empty())
	  {
		throw std::runtime_error("Invalid parameter: " + first);
	  }
	}
	catch (const NetworkParamError& e)
	{
	  std::stringstream ss;
	  ss << "Error at line " << e.LineNo() << " of job file " << StripDir(fileName) << ": " << e.Error();
	  throw std::runtime_error(ss.str());
	}
	catch (const std::exception& e)
	{
	  std::stringstream ss;
	  ss << "Error at line " << lineNo << " of job file " << StripDir(fileName) << ": " << e.what();
	  throw std::runtime_error(ss.str());
	}
  }
}

void Trainer::TrainAll()
{
  std::string saveDir = Utils::GetEnv("FISHNET_SAVE_DIR");
  if (saveDir.back() != PATH_SEPARATOR)
	saveDir += PATH_SEPARATOR;

  for (auto& job : _jobs)
  {
	job->Run(saveDir);
	delete job.release();
  }
  LOG(Info) << "All training completed.";
}

static const std::string& GetParam(const std::vector<std::string>& params, int index)
{
  static std::string empty;
  return index != -1 ? params[index] : empty;
}

std::unique_ptr<FeedForwardNetwork> Trainer::LoadNetwork(const std::string& name, std::ifstream& is,
  int& lineNo, const ImageSet& imageSet, double learningRate, double weightDecay)
{
  std::string line;
  std::getline(is, line);
  ++lineNo;

  int activationCol = -1;
  int dropoutCol = -1;
  int filterCountCol = -1;
  int filterSizeCol = -1;
  int leakinessCol = -1;
  int layerCol = -1;
  int layerSizeCol = -1;
  int paddingCol = -1;
  int strideCol = -1;

  StringUtils::ToLower(line);
  std::vector<std::string> fields;
  StringUtils::SplitCSV(fields, line);
  std::set<std::string> existingParams;
  for (int i = 0; i < fields.size(); ++i)
  {
	const std::string& paramName = fields[i];
	if (!existingParams.insert(paramName).second)
	  throw NetworkParamError("Duplicate network parameter: " + paramName, lineNo);

	if (paramName == "activation")
	  activationCol = i;
	else if (paramName == "dropout")
	  dropoutCol = i;
	else if (paramName == "filter count")
	  filterCountCol = i;
	else if (paramName == "filter size")
	  filterSizeCol = i;
	else if (paramName == "leakiness")
	  leakinessCol = i;
	else if (paramName == "layer")
	  layerCol = i;
	else if (paramName == "layer size")
	  layerSizeCol = i;
	else if (paramName == "padding")
	  paddingCol = i;
	else if (paramName == "stride")
	  strideCol = i;
	else if (!paramName.empty())
	  throw NetworkParamError("Invalid network parameter: " + paramName, lineNo);
  }

  if (activationCol == -1)
	throw NetworkParamError("Compulsory network parameter \"activation\" is missing.", lineNo);
  if (layerCol == -1)
	throw NetworkParamError("Compulsory network parameter \"layer\" is missing.", lineNo);

  auto network = std::make_unique<FeedForwardNetwork>(name, imageSet.Channels(), imageSet.Height(), imageSet.Width(),
	std::make_unique<CrossEntropyCostFunction>(), _threadCount, 0, learningRate, weightDecay);

  while (!is.eof())
  {
	std::getline(is, line);
	++lineNo;
	if (line.empty())
	  break;
	if (line.front() == '#')
	  continue;
	try
	{
	  StringUtils::ToLower(line);
	  StringUtils::SplitCSV(fields, line);
	  if (fields.front().empty())
		break;
	  if (fields[layerCol] == "max pooling")
	  {
		network->AddMaxPoolingLayer();
	  }
	  else
	  {
		std::unique_ptr<ActivationFunction> activationFunction;
		const std::string& activation = GetParam(fields, activationCol);
		if (activation.empty() || activation == "sigmoid")
		{
		  activationFunction = std::make_unique<Sigmoid>();
		}
		else if (activation == "leaky relu")
		{
		  double leakiness = 0.01;
		  const auto& param = GetParam(fields, leakinessCol);
		  if (!param.empty())
			leakiness = std::stod(param);
		  if (leakiness <= 0.0 || leakiness >= 1.0)
			throw std::runtime_error("Leakiness must be greater than 0 and less than 1.");
		  activationFunction = std::make_unique<LeakyReLU>(leakiness);
		}
		else if (activation == "relu")
		{
		  activationFunction = std::make_unique<ReLU>();
		}
		else if (activation == "tanh")
		{
		  activationFunction = std::make_unique<TanH>();
		}
		else
		{
		  throw std::runtime_error("Invalid activation function: " + activation);
		}

		if (fields[layerCol] == "fully connected")
		{
		  const std::string& sizeParam = GetParam(fields, layerSizeCol);
		  if (sizeParam.empty())
			throw std::runtime_error("Compulsory parameter \"layer size\" is missing.");
		  int layerSize = std::stoi(sizeParam);
		  if (layerSize == 0)
			throw std::runtime_error("Layer size must be at least 1.");

		  double dropout = 0.0;
		  const std::string& dropoutParam = GetParam(fields, dropoutCol);
		  if (!dropoutParam.empty())
		  {
			dropout = std::stod(dropoutParam);
			if (dropout >= 1.0 || dropout < 0.0)
			  throw std::runtime_error("Dropout must be greater than or equal to 0 and less than 1.");
		  }
		  network->AddFullyConnectedLayer(layerSize, std::move(activationFunction), 1.0 - dropout);
		}
		else if (fields[layerCol] == "convolutional")
		{
		  const std::string& countParam = GetParam(fields, filterCountCol);
		  if (countParam.empty())
			throw std::runtime_error("Compulsory parameter \"filter count\" is missing.");
		  int filterCount = std::stoi(countParam);
		  if (filterCount < 1)
			throw std::runtime_error("Filter count must be at least 1.");

		  const std::string& sizeParam = GetParam(fields, filterSizeCol);
		  if (sizeParam.empty())
			throw std::runtime_error("Compulsory parameter \"filter size\" is missing.");
		  int filterSize = std::stoi(sizeParam);
		  if (filterSize < 1)
			throw std::runtime_error("Filter size must be strictly positive.");
		  if (filterSize >= (int)imageSet.Height() || filterSize >= (int)imageSet.Width())
			throw std::runtime_error("Filter must be smaller than the image dimensions.");

		  int zeroPadding = 0;
		  const std::string& paddingParam = GetParam(fields, paddingCol);
		  if (!paddingParam.empty())
		  {
			zeroPadding = std::stoi(paddingParam);
			if (zeroPadding < 0)
			  throw std::runtime_error("Padding cannot be negative.");
		  }

		  int stride = 1;
		  const std::string& strideParam = GetParam(fields, strideCol);
		  if (!strideParam.empty())
		  {
			stride = std::stoi(strideParam);
			if (stride < 1)
			  throw std::runtime_error("Stride must be at least 1.");
		  }
		  network->AddConvolutionalLayer(filterCount, filterSize, stride, zeroPadding, std::move(activationFunction));
		}
		else
		{
		  throw std::runtime_error("Invalid layer type: " + fields[layerCol]);
		}
	  }
	}
	catch (const std::exception& e)
	{
	  throw NetworkParamError(e.what(), lineNo);
	}
  }

  if (network->Layers().empty())
	throw NetworkParamError("Network does not contain any layers.", lineNo);
  auto output = dynamic_cast<FullyConnectedLayer*>(network->Layers().back().get());
  if (!output || output->OutputColumns() != imageSet.OneHotCategories().size())
	throw NetworkParamError("The output layer of the network must be a fully connected layer with one neuron for each category in the data set.",
	  lineNo - 1);
  if (output->ActivationFunction()->Type() != ActivationFunction::Types::Sigmoid)
	throw NetworkParamError("The output layer of the network must use sigmoid activation.", lineNo - 1);
  return network;
}
