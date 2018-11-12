#include "stdafx.h"
#include "ActivationFunction.h"
#include "Tensor.h"

void ActivationFunction::Save(std::ofstream& os) const
{
  os.put((char)Type());
}

std::unique_ptr<ActivationFunction> ActivationFunction::Load(std::ifstream& is)
{
  auto type = static_cast<Types>(is.get());
  switch(type)
  {
	case Types::None:
	  return nullptr;
	case Types::ReLU:
	  return std::make_unique<ReLU>();
	case Types::LeakyReLU:
	{
	  double leakiness;
	  is.read((char*)&leakiness, sizeof(double));
	  return std::make_unique<LeakyReLU>(leakiness);
	}
	case Types::Sigmoid:
	  return std::make_unique<Sigmoid>();
	case Types::TanH:
	  return std::make_unique<TanH>();
	default:
	  throw std::runtime_error("Unrecognized activation function code: " + std::to_string((int)type));
  }
}

void ReLU::Apply(Tensor& tensor) const noexcept
{
  double* v = tensor.Elements();
  const double* end = v + tensor.Size();
  while (v < end)
  {
	if (*v < 0.0)
	  *v = 0.0;
	++v;
  }
}

void ReLU::ApplyDerivative(Tensor& input, Tensor& output) const
{
#ifdef _DEBUG
  if (input.Size() != output.Size())
	throw std::runtime_error("Parameters to ApplyDerivative must be the same size.");
#endif
  const double* in = input.Elements();
  double* out = output.Elements();
  const double* end = in + input.Size();
  while (in < end)
  {
	if (*in <= 0.0)
	  *out = 0.0;
	else
	  *out = 1.0;
	++in;
	++out;
  }
}

void LeakyReLU::Apply(Tensor& tensor) const noexcept
{
  double* v = tensor.Elements();
  const double* end = v + tensor.Size();
  while (v < end)
  {
	if (*v < 0.0)
	  *v *= _leakiness;
	++v;
  }
}

void LeakyReLU::ApplyDerivative(Tensor& input, Tensor& output) const
{
#ifdef _DEBUG
  if (input.Size() != output.Size())
	throw std::runtime_error("Parameters to ApplyDerivative must be the same size.");
#endif
  const double* in = input.Elements();
  double* out = output.Elements();
  const double* end = in + input.Size();
  while (in < end)
  {
	if (*in <= 0.0)
	  *out = _leakiness;
	else
	  *out = 1.0;
	++in;
	++out;
  }
}

void LeakyReLU::Save(std::ofstream& os) const
{
  os.put((char)Types::LeakyReLU);
  os.write((const char*)&_leakiness, sizeof(double));
}

void Sigmoid::Apply(Tensor& tensor) const noexcept
{
  double* v = tensor.Elements();
  const double* end = v + tensor.Size();
  while (v < end)
  {
	*v = 1.0 / (1.0 + exp(-(*v)));
	++v;
  }
}

void Sigmoid::ApplyDerivative(Tensor& input, Tensor& output) const
{
#ifdef _DEBUG
  if (input.Size() != output.Size())
	throw std::runtime_error("Parameters to ApplyDerivative must be the same size.");
#endif
  const double* in = input.Elements();
  double* out = output.Elements();
  const double* end = in + input.Size();
  while (in < end)
  {
	double sig = 1.0 / (1.0 + exp(-(*in)));
	*out = sig * (1.0 - sig);
	++in;
	++out;
  }
}

void TanH::Apply(Tensor& tensor) const noexcept
{
  double* v = tensor.Elements();
  const double* end = v + tensor.Size();
  while (v < end)
  {
	*v = 2.0 / (1.0 + exp(-2.0 * *v)) - 1.0;
	++v;
  }
}

void TanH::ApplyDerivative(Tensor& input, Tensor& output) const
{
#ifdef _DEBUG
  if (input.Size() != output.Size())
	throw std::runtime_error("Parameters to ApplyDerivative must be the same size.");
#endif
  const double* in = input.Elements();
  double* out = output.Elements();
  const double* end = in + input.Size();
  while (in < end)
  {
	double tanh = 2.0 / (1.0 + exp(-2.0 * *in)) - 1.0;
	*out = 1.0 - (tanh * tanh);
	++in;
	++out;
  }
}
