#pragma once

class Tensor;

class ActivationFunction
{
public:
  enum class Types { None = 0, ReLU = 1, LeakyReLU = 2, Sigmoid = 3, TanH = 4 };

  virtual ~ActivationFunction() {}
  virtual Types Type() const noexcept = 0;
  virtual const char* Name() const noexcept = 0;
  virtual std::string Description() const = 0;
  virtual void Apply(Tensor&) const noexcept = 0;
  virtual void ApplyDerivative(Tensor& input, Tensor& output) const = 0;
  virtual void Save(std::ofstream&) const;
  static std::unique_ptr<ActivationFunction> Load(std::ifstream&);
};

class ReLU : public ActivationFunction
{
public:
  virtual Types Type() const noexcept override { return Types::ReLU; }
  virtual const char* Name() const noexcept override { return "ReLU"; }
  virtual std::string Description() const noexcept override { return "ReLU"; }
  virtual void Apply(Tensor&) const noexcept override;
  virtual void ApplyDerivative(Tensor& input, Tensor& output) const override;
};

class LeakyReLU : public ActivationFunction
{
public:
  LeakyReLU(double leakiness)
	: _leakiness(leakiness) {}
  virtual Types Type() const noexcept override { return Types::LeakyReLU; }
  virtual const char* Name() const noexcept override { return "Leaky ReLU"; }
  virtual std::string Description() const override
  {
	return std::string("Leaky ReLU, leakiness ") + std::to_string(_leakiness);
  }
  double Leakiness() const noexcept { return _leakiness; }
  virtual void Apply(Tensor&) const noexcept override;
  virtual void ApplyDerivative(Tensor& input, Tensor& output) const override;
  virtual void Save(std::ofstream&) const override;
private:
  double _leakiness;
};

class Sigmoid : public ActivationFunction
{
public:
  virtual Types Type() const noexcept override { return Types::Sigmoid; }
  virtual const char* Name() const noexcept override { return "Sigmoid"; }
  virtual std::string Description() const override { return "Sigmoid"; }
  virtual void Apply(Tensor&) const noexcept override;
  virtual void ApplyDerivative(Tensor& input, Tensor& output) const override;
};

class TanH : public ActivationFunction
{
public:
  virtual Types Type() const noexcept override { return Types::TanH; }
  virtual const char* Name() const noexcept override { return "TanH"; }
  virtual std::string Description() const override { return "TanH"; }
  virtual void Apply(Tensor&) const noexcept override;
  virtual void ApplyDerivative(Tensor& input, Tensor& output) const override;
};