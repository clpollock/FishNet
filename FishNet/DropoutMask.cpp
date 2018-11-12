#include "stdafx.h"
#include "DropoutMask.h"

void DropoutMask::Randomize()
{
  static std::default_random_engine generator(static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count()));
  const bool* end = _mask.get() + _size;
  for (bool* b = _mask.get(); b != end; ++b)
	*b = _distribution(generator);
}
