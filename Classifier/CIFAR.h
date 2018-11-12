#pragma once

#include <ImageSet.h>

extern std::unique_ptr<ImageSet> LoadCIFAR10(const std::string& dataDir, bool dry);
