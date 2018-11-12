#pragma once

// This code is a heavily modified version of a MNIST loader by Nuri Park - https://github.com/projectgalateia/mnist

#include <ImageSet.h>

extern std::unique_ptr<ImageSet> LoadMNIST(const std::string& dataDir, bool dry);
