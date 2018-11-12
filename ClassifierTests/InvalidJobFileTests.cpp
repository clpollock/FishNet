#include "stdafx.h"
#include "CppUnitTest.h"
#include "Trainer.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

std::string TestFile(const char* file)
{
  std::string testDir(__FILE__);
  const size_t i = testDir.rfind('\\');
  if (i != std::string::npos)
      return testDir.substr(0, i + 1) + file;
  return file;
}

namespace ClassifierTests
{		
	TEST_CLASS(InvalidJobFileTests)
	{
	public:
		TEST_METHOD(JobWithBadDataSet)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadDataSet.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>("Error at line 1 of job file BadDataSet.csv: Unknown image set death. The supported "
			  "image sets are cifar-10, mnist, emotions, face-directions, people, sunglasses, and directions-sunglasses.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithBadDecay)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadDecay.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>(
			  "Error at line 3 of job file BadDecay.csv: Learning rate decay must be greater than or equal to 0 and less than 1.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithBadEpochs)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadEpochs.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>("Error at line 5 of job file BadEpochs.csv: Number of epochs must be at least 1.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithBadLearningRate)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadLearningRate.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>("Error at line 2 of job file BadLearningRate.csv: Learning rate must be greater than 0.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithBadMiniBatch)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadMiniBatch.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>("Error at line 4 of job file BadMiniBatch.csv: Minibatch size must be at least 1.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithEmptyNetwork)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\EmptyNetwork.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>(
			  "Error at line 7 of job file EmptyNetwork.csv: Compulsory network parameter \"activation\" is missing.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithBadActivation)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadActivation.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>(
			  "Error at line 8 of job file BadActivation.csv: Invalid activation function: rlu", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithBadLayerType)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadLayerType.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>("Error at line 8 of job file BadLayerType.csv: Invalid layer type: conovlutional", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithBadLayerSize)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadLayerSize.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>(
			  "Error at line 12 of job file BadLayerSize.csv: Layer size must be at least 1.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithBadFilterCount)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadFilterCount.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>(
			  "Error at line 10 of job file BadFilterCount.csv: Filter count must be at least 1.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithBadFilterSize)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadFilterSize.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>(
			  "Error at line 8 of job file BadFilterSize.csv: Filter must be smaller than the image dimensions.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithBadStride)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadStride.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>(
			  "Error at line 8 of job file BadStride.csv: Stride must be at least 1.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithBadPadding)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadPadding.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>(
			  "Error at line 8 of job file BadPadding.csv: Zero padding must be less than the size of the filter.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithBadDropout)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadDropout.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>(
			  "Error at line 12 of job file BadDropout.csv: Dropout must be greater than or equal to 0 and less than 1.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithBadLeakiness)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\BadLeakiness.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>(
			  "Error at line 12 of job file BadLeakiness.csv: Leakiness must be greater than 0 and less than 1.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithConvolutionalLayerAfterFullyConnected)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\InvalidConvolutional.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>(
			  "Error at line 11 of job file InvalidConvolutional.csv: A convolutional layer cannot follow a fully connected layer.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithInvalidMaxPoolingLayer)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\InvalidMaxPoolingLayer.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>(
			  "Error at line 12 of job file InvalidMaxPoolingLayer.csv: A max pooling layer cannot follow a fully connected layer.", e.what());
		  }
		  Assert::IsTrue(caught);
		}

		TEST_METHOD(JobWithInvalidOutputLayer)
		{
		  bool caught = false;
		  try
		  {
			ImageSetLoader loader(true);
			Trainer trainer(loader, 1);
			trainer.LoadJobList(TestFile("BadJobs\\InvalidOutputLayer.csv"));
		  }
		  catch (const std::exception& e)
		  {
			caught = true;
			Assert::AreEqual<std::string>("Error at line 13 of job file InvalidOutputLayer.csv: The output layer of the "
			  "network must be a fully connected layer with one neuron for each category in the data set.", e.what());
		  }
		  Assert::IsTrue(caught);
		}
	};
}