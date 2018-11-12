#pragma once

class Tensor
{
public:
  Tensor(uint32_t hyperplanes, uint32_t planes, uint32_t rows, uint32_t columns)
	: _elements(std::make_unique<double[]>(hyperplanes * planes * rows * columns)),
	  _hyperplanes(hyperplanes), _planes(planes), _rows(rows), _columns(columns),
	  _planeSize(rows * columns),
	  _hyperplaneSize(_planeSize * planes),
	  _size(_hyperplaneSize * hyperplanes)
  {
	memset(_elements.get(), 0, sizeof(double) * _size);
  }
  Tensor(uint32_t planes, uint32_t rows, uint32_t columns)
	: Tensor(1, planes, rows, columns) {}
  Tensor(uint32_t rows, uint32_t columns)
	: Tensor(1, 1, rows, columns) {}
  Tensor(uint32_t size)
	: Tensor(1, 1, 1, size) {}

  Tensor(std::unique_ptr<double[]>&& elements, uint32_t hyperplanes, uint32_t planes, uint32_t rows, uint32_t columns)
	: _elements(std::move(elements)),
	  _hyperplanes(hyperplanes), _planes(planes), _rows(rows), _columns(columns),
	  _planeSize(rows * columns),
	  _hyperplaneSize(_planeSize * planes),
	  _size(_hyperplaneSize * hyperplanes) {}
  Tensor(std::unique_ptr<double[]>&& elements, uint32_t planes, uint32_t rows, uint32_t columns)
	: Tensor(std::move(elements), 1, planes, rows, columns) {}
  Tensor(std::unique_ptr<double[]>&& elements, uint32_t rows, uint32_t columns)
	: Tensor(std::move(elements), 1, 1, rows, columns) {}
  Tensor(std::unique_ptr<double[]>&& elements, uint32_t size)
	: Tensor(std::move(elements), 1, 1, 1, size) {}
  // Constructors for initializer_list
  Tensor(const std::initializer_list<double>& elements, uint32_t hyperplanes, uint32_t planes, uint32_t rows, uint32_t columns);
  Tensor(const std::initializer_list<double>& elements, uint32_t planes, uint32_t rows, uint32_t columns)
	: Tensor(std::move(elements), 1, planes, rows, columns) {}
  Tensor(const std::initializer_list<double>& elements, uint32_t rows, uint32_t columns)
    : Tensor(std::move(elements), 1, 1, rows, columns) {}
  Tensor(const std::initializer_list<double>& elements)
	: Tensor(std::move(elements), 1, 1, 1, static_cast<uint32_t>(elements.size())) {}
  // Copy and move constructors
  Tensor(const Tensor&);
  Tensor(Tensor&& that) noexcept
	: _elements(std::move(that._elements)),
	  _hyperplanes(that._hyperplanes),
	  _planes(that._planes),
	  _rows(that._rows),
	  _columns(that._columns),
	  _planeSize(that._planeSize),
	  _hyperplaneSize(that._hyperplaneSize),
	  _size(that._size)
  {
	that._hyperplanes = 0;
	that._planes = 0;
	that._rows = 0;
	that._columns = 0;
	that._planeSize = 0;
	that._hyperplaneSize = 0;
	that._size = 0;
  }
  ~Tensor() {}
  void SetAllToZero()
  {
	memset(_elements.get(), 0, sizeof(double) * _size);
  }
  void Fill(double value)
  {
	double* end = _elements.get() + _size;
	for (double* v = _elements.get(); v < end; ++v)
	  *v = value;
  }
  Tensor& operator=(const Tensor&);
  bool DimensionsMatch(const Tensor& other) const
  {
	return _hyperplanes == other._hyperplanes && _planes == other._planes && _rows == other._rows && _columns == other._columns;
  }
  void ComponentWiseAdd(const Tensor& other, Tensor& result) const;
  void ComponentWiseAdd(const Tensor& other);
  void ComponentWiseSubtract(const Tensor& other, Tensor& result) const;
  void ComponentWiseSubtract(const Tensor& other);
  void ComponentWiseMultiply(const Tensor& other, Tensor& result) const;
  void ComponentWiseMultiply(const Tensor& other);
  uint32_t HighestValueIndex() const;
  double Get(uint32_t i) const
  {
#ifdef _DEBUG
	if (i >= _size)
	  throw std::runtime_error("Index out of bounds for Tensor::Get.");
#endif
	return _elements[i];
  }
  void Set(uint32_t i, double value)
  {
#ifdef _DEBUG
	if (i >= _size)
	  throw std::runtime_error("Index out of bounds for Tensor::Set.");
#endif
	_elements[i] = value;
  }
  double Get(uint32_t row, uint32_t column) const
  {
#ifdef _DEBUG
	if (_hyperplanes > 1 || _planes > 1)
	{
	  std::ostringstream error;
	  error << "Attemped to do a 2-dimensional get on a higher dimensional tensor.";
	  throw std::runtime_error(error.str());
	}
	if (row > _rows)
	  throw std::runtime_error("Row is out of bounds for Tensor::Get.");
	if (column > _columns)
	  throw std::runtime_error("Column is out of bounds for Tensor::Get.");
#endif
	return _elements[(_columns * row) + column];
  }
  void Set(uint32_t row, uint32_t column, double value)
  {
#ifdef _DEBUG
	if (_hyperplanes > 1 || _planes > 1)
	{
	  std::ostringstream error;
	  error << "Attemped to do a 2-dimensional set on a higher dimensional tensor.";
	  throw std::runtime_error(error.str());
	}
	if (row > _rows)
	  throw std::runtime_error("Row is out of bounds for Tensor::Set.");
	if (column > _columns)
	  throw std::runtime_error("Column is out of bounds for Tensor::Set.");
#endif
	_elements[(_columns * row) + column] = value;
  }
  double Get(uint32_t plane, uint32_t row, uint32_t column) const
  {
#ifdef _DEBUG
	if (_hyperplanes > 1)
	{
	  std::ostringstream error;
	  error << "Attemped to do a 3-dimensional get on a 4 dimensional tensor.";
	  throw std::runtime_error(error.str());
	}
	if (plane > _planes)
	  throw std::runtime_error("Channel is out of bounds for Tensor::Get.");
	if (row > _rows)
	  throw std::runtime_error("Row is out of bounds for Tensor::Get.");
	if (column > _columns)
	  throw std::runtime_error("Column is out of bounds for Tensor::Get.");
#endif
	return _elements[(_planeSize * plane) + (_columns * row) + column];
  }
  void Set(uint32_t plane, uint32_t row, uint32_t column, double value)
  {
#ifdef _DEBUG
	if (_hyperplanes > 1)
	{
	  std::ostringstream error;
	  error << "Attemped to do a 3-dimensional set on a 4 dimensional tensor.";
	  throw std::runtime_error(error.str());
	}
	if (plane > _planes)
	  throw std::runtime_error("Channel is out of bounds for Tensor::Set.");
	if (row > _rows)
	  throw std::runtime_error("Row is out of bounds for Tensor::Set.");
	if (column > _columns)
	  throw std::runtime_error("Column is out of bounds for Tensor::Set.");
#endif
	_elements[(_planeSize * plane) + (_columns * row) + column] = value;
  }
  double Get(uint32_t hyperplane, uint32_t plane, uint32_t row, uint32_t column) const
  {
#ifdef _DEBUG
	if (hyperplane > _hyperplanes)
	  throw std::runtime_error("4th dimension is out of bounds for Tensor::Get.");
	if (plane > _planes)
	  throw std::runtime_error("Channel is out of bounds for Tensor::Get.");
	if (row > _rows)
	  throw std::runtime_error("Row is out of bounds for Tensor::Get.");
	if (column > _columns)
	  throw std::runtime_error("Column is out of bounds for Tensor::Get.");
#endif
	return _elements[(_hyperplaneSize * hyperplane) + (_planeSize * plane) + (_columns * row) + column];
  }
  double* ElementAddress(uint32_t i)
  {
#ifdef _DEBUG
	if (i >= _size)
	  throw std::runtime_error("Index out of bounds for Tensor::ElementAddress.");
#endif
	return _elements.get() + i;
  }
  double* ElementAddress(uint32_t row, uint32_t column)
  {
#ifdef _DEBUG
	if (_hyperplanes > 1 || _planes > 1)
	{
	  std::ostringstream error;
	  error << "Attemped to do a 2-dimensional operation on a higher dimensional tensor.";
	  throw std::runtime_error(error.str());
	}
	if (row > _rows)
	  throw std::runtime_error("Row is out of bounds for Tensor::ElementAddress.");
	if (column > _columns)
	  throw std::runtime_error("Column is out of bounds for Tensor::ElementAddress.");
#endif
	return _elements.get() + (_columns * row) + column;
  }
  double* ElementAddress(uint32_t plane, uint32_t row, uint32_t column)
  {
#ifdef _DEBUG
	if (_hyperplanes > 1)
	{
	  std::ostringstream error;
	  error << "Attemped to do a 3-dimensional operation on a 4 dimensional tensor.";
	  throw std::runtime_error(error.str());
	}
	if (plane > _planes)
	  throw std::runtime_error("Channel is out of bounds for Tensor::Get.");
	if (row > _rows)
	  throw std::runtime_error("Row is out of bounds for Tensor::Get.");
	if (column > _columns)
	  throw std::runtime_error("Column is out of bounds for Tensor::Get.");
#endif
	return _elements.get() + (_planeSize * plane) + (_columns * row) + column;
  }
  double* ElementAddress(uint32_t hyperPlane, uint32_t plane, uint32_t row, uint32_t column)
  {
#ifdef _DEBUG
	if (hyperPlane > _hyperplanes)
	  throw std::runtime_error("Hyperplane is out of bounds for Tensor::Get.");
	if (plane > _planes)
	  throw std::runtime_error("Plane is out of bounds for Tensor::Get.");
	if (row > _rows)
	  throw std::runtime_error("Row is out of bounds for Tensor::Get.");
	if (column > _columns)
	  throw std::runtime_error("Column is out of bounds for Tensor::Get.");
#endif
	return _elements.get() + (_hyperplaneSize * hyperPlane) + (_planeSize * plane) + (_columns * row) + column;
  }
  const double* ElementAddress(uint32_t i) const
  {
	// Ugly but harmless little hack.
	return const_cast<Tensor*>(this)->ElementAddress(i);
  }
  const double* ElementAddress(uint32_t row, uint32_t column) const
  {
	return const_cast<Tensor*>(this)->ElementAddress(row, column);
  }
  const double* ElementAddress(uint32_t plane, uint32_t row, uint32_t column) const
  {
	return const_cast<Tensor*>(this)->ElementAddress(plane, row, column);
  }
  double* Elements() const { return _elements.get(); }
  uint32_t Hyperplanes() const { return _hyperplanes; }
  uint32_t Planes() const { return _planes; }
  uint32_t Rows() const { return _rows; }
  uint32_t Columns() const { return _columns; }
  uint32_t PlaneSize() const { return _planeSize; }
  uint32_t HyperplaneSize() const { return _hyperplaneSize; }
  uint32_t Size() const { return _size; }
  void GetStatistics(double& maxWeight, double& minWeight, double& avgWeight) const;
  void Save(std::ofstream&);
  static std::unique_ptr<Tensor> Load(std::ifstream&);
private:
  std::unique_ptr<double[]> _elements;
  uint32_t _hyperplanes;
  uint32_t _planes;
  uint32_t _rows;
  uint32_t _columns;
  uint32_t _planeSize;
  uint32_t _hyperplaneSize;
  uint32_t _size;
};

using TensorPtr = std::unique_ptr<Tensor>;

extern std::ostream& operator<<(std::ostream& os, const Tensor& t);
