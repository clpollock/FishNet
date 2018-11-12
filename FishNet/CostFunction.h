#pragma once

class Tensor;

class CostFunction
{
public:
  enum class Types { CrossEntropy = 1 };
  virtual ~CostFunction() {}
  virtual Types Type() const = 0;
  virtual double TotalCost(const Tensor& outputActivations, const Tensor& targetActivations) const = 0;
  virtual void Derivatives(const Tensor& outputActivations, const Tensor& targetActivations, Tensor& result) const = 0;
};

class CrossEntropyCostFunction : public CostFunction
{
public:
  virtual Types Type() const override { return Types::CrossEntropy; }
  virtual double TotalCost(const Tensor& outputActivations, const Tensor& targetActivations) const override;
  virtual void Derivatives(const Tensor& outputActivations, const Tensor& targetActivations, Tensor& result) const override;
};