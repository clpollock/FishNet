// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#ifdef _WIN32
#include <tchar.h>
#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#else
#include <string.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdio.h>
#include <thread>
#include <vector>

#include <Log.h>
#include <Utils.h>
#include <StringUtils.h>
#include <CostFunction.h>
#include <FeedForwardNetwork.h>
#include <Layer.h>
