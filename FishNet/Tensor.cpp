#include "stdafx.h"
#include "Tensor.h"

Tensor::Tensor(const std::initializer_list<double>& elements, uint32_t hyperplanes, uint32_t planes, uint32_t rows, uint32_t columns)
  : _elements(std::make_unique<double[]>(elements.size())),
	_hyperplanes(hyperplanes), _planes(planes), _rows(rows), _columns(columns),
	_planeSize(rows * columns),
	_hyperplaneSize(_planeSize * planes),
	_size(_hyperplaneSize * hyperplanes)
{
  if (elements.size() != _size)
	throw std::runtime_error("Size of initializer list does not match dimensions of Tensor.");
  memcpy(_elements.get(), elements.begin(), elements.size() * sizeof(double));
}

Tensor::Tensor(const Tensor& that)
  : _elements(std::make_unique<double[]>(that._size)),
	_hyperplanes(that._hyperplanes), _planes(that._planes), _rows(that._rows), _columns(that._columns),
	_planeSize(that._planeSize),
	_hyperplaneSize(that._hyperplaneSize),
	_size(that._size)
{
  memcpy(_elements.get(), that._elements.get(), sizeof(double) * _size);
}

Tensor& Tensor::operator=(const Tensor& that)
{
  _hyperplanes = that._hyperplanes;
  _planes = that._planes;
  _rows = that._rows;
  _columns = that._columns;
  _planeSize = that._planeSize;
  _hyperplaneSize = that._hyperplaneSize;
  if (_size != that._size)
  {
	_size = that._size;
	_elements.reset(new double[_size]);
  }
  memcpy(_elements.get(), that._elements.get(), sizeof(double) * _size);
  return *this;
}

void Tensor::ComponentWiseAdd(const Tensor& other, Tensor& result) const
{
#ifdef _DEBUG
  if (_size != other._size || _size != result._size)
	throw std::runtime_error("All parameters to PairwiseSubtract must be the same size.");
#endif
  double* v1 = _elements.get();
  const double* v2 = other._elements.get();
  double* r = result._elements.get();
  const double* end = r + result._size;
  while (r < end)
  {
	*r = *v1 + *v2;
	++r;
	++v1;
	++v2;
  }
}

void Tensor::ComponentWiseAdd(const Tensor& other)
{
#ifdef _DEBUG
  if (_size != other._size)
	throw std::runtime_error("The parameters to PairwiseSubtract must be the same size.");
#endif
  double* v1 = _elements.get();
  const double* v2 = other._elements.get();
  const double* end = v1 + _size;
  while (v1 < end)
  {
	*v1 += *v2;
	++v1;
	++v2;
  }
}

void Tensor::ComponentWiseSubtract(const Tensor& other, Tensor& result) const
{
#ifdef _DEBUG
  if (_size != other._size || _size != result._size)
	throw std::runtime_error("All parameters to PairwiseSubtract must be the same size.");
#endif
  double* v1 = _elements.get();
  const double* v2 = other._elements.get();
  double* r = result._elements.get();
  const double* end = r + result._size;
  while (r < end)
  {
	*r = *v1 - *v2;
	++r;
	++v1;
	++v2;
  }
}

void Tensor::ComponentWiseSubtract(const Tensor& other)
{
#ifdef _DEBUG
  if (_size != other._size)
	throw std::runtime_error("The parameters to PairwiseSubtract must be the same size.");
#endif
  double* v1 = _elements.get();
  const double* v2 = other._elements.get();
  const double* end = v1 + _size;
  while (v1 < end)
  {
	*v1 -= *v2;
	++v1;
	++v2;
  }
}

void Tensor::ComponentWiseMultiply(const Tensor& other, Tensor& result) const
{
#ifdef _DEBUG
  if (_size != other._size || _size != result._size)
	throw std::runtime_error("All parameters to PairwiseSubtract must be the same size.");
#endif
  double* v1 = _elements.get();
  const double* v2 = other._elements.get();
  double* r = result._elements.get();
  const double* end = r + result._size;
  while (r < end)
  {
	*r = *v1 * *v2;
	++r;
	++v1;
	++v2;
  }
}

void Tensor::ComponentWiseMultiply(const Tensor& other)
{
#ifdef _DEBUG
  if (_size != other._size)
	throw std::runtime_error("The parameters to PairwiseSubtract must be the same size.");
#endif
  double* v1 = _elements.get();
  const double* v2 = other._elements.get();
  const double* end = v1 + _size;
  while (v1 < end)
  {
	*v1 *= *v2;
	++v1;
	++v2;
  }
}

uint32_t Tensor::HighestValueIndex() const
{
#ifdef _DEBUG
  if (_size == 0)
	throw std::runtime_error("HighestValueIndex called on empty Tensor.");
#endif 
  const double* v = _elements.get();
  const double* end = v + _size;
  const double* highestAddress = v;
  double highest = *highestAddress;
  while (++v < end)
  {
	if (*v > highest)
	{
	  highestAddress = v;
	  highest = *v;
	}
  }

  return static_cast<uint32_t>(highestAddress - _elements.get());
}

void Tensor::GetStatistics(double& maxWeight, double& minWeight, double& avgWeight) const
{
  maxWeight = std::numeric_limits<double>::min();
  minWeight = std::numeric_limits<double>::max();
  avgWeight = 0.0;
  const double* end = _elements.get() + _size;
  for (const double* w = _elements.get(); w != end; ++w)
  {
	if (*w < minWeight)
	  minWeight = *w;
	if (*w > maxWeight)
	  maxWeight = *w;
	avgWeight += *w;
  }
  avgWeight /= (double)_size;
}

void Tensor::Save(std::ofstream& os)
{
  os.write((const char*)&_hyperplanes, 4);
  os.write((const char*)&_planes, 4);
  os.write((const char*)&_rows, 4);
  os.write((const char*)&_columns, 4);
  os.write((const char*)_elements.get(), _size * sizeof(double));
}

std::unique_ptr<Tensor> Tensor::Load(std::ifstream& is)
{
  uint32_t hyperplanes;
  uint32_t planes;
  uint32_t rows;
  uint32_t columns;
  is.read((char*)&hyperplanes, 4);
  is.read((char*)&planes, 4);
  is.read((char*)&rows, 4);
  is.read((char*)&columns, 4);
  uint32_t size = hyperplanes * planes * rows * columns;
  auto elements = std::make_unique<double[]>(size);
  is.read((char*)elements.get(), size * sizeof(double));
  return std::make_unique<Tensor>(std::move(elements), hyperplanes, planes, rows, columns);
}

std::ostream& operator<<(std::ostream& os, const Tensor& t)
{
  if (t.Hyperplanes() > 1)
	os << t.Hyperplanes() << 'x' << t.Planes() << 'x' << t.Rows() << 'x';
  else if (t.Planes() > 1)
	os << t.Planes() << 'x' << t.Rows() << 'x';
  else if (t.Rows() > 1)
	os << t.Rows() << 'x';
  os << t.Columns();
  return os;
}
