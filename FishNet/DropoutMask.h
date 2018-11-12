#pragma once

class DropoutMask
{
public:
  DropoutMask(double keepProbability, uint32_t size)
	: _distribution(keepProbability),
	  _mask(std::make_unique<bool[]>(size)),
	  _size(size) {}
	// Create a DropoutMask with a fixed pattern. This should only be used for testing.
  DropoutMask(const std::initializer_list<bool>& elements)
	: _mask(std::make_unique<bool[]>(elements.size())),
	  _size(static_cast<uint32_t>(elements.size()))
  {
	memcpy(_mask.get(), elements.begin(), elements.size() * sizeof(bool));
  }
  void Randomize();
  const bool* Begin() const { return _mask.get(); }
  uint32_t Size() const { return _size; }
  bool Get(uint32_t i) const
  {
#ifdef _DEBUG
	if (i >= _size)
	  throw std::runtime_error("Index out of bounds for DropoutMask::Get.");
#endif
	return _mask[i];
  }
private:
  std::bernoulli_distribution _distribution;
  std::unique_ptr<bool[]> _mask;
  uint32_t _size;
};

using DropoutMaskPtr = std::unique_ptr<DropoutMask>;
