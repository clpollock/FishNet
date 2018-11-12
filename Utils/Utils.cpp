#include "stdafx.h"
#include "Utils.h"

std::string Utils::GetEnv(const char* variable)
{
#ifdef _WIN32
  char* value;
  if (_dupenv_s(&value, nullptr, variable) || !value)
	throw std::runtime_error(std::string("Couldn't read environment variable ") + variable);
  std::string s(value);
  delete[] value;
#else
  const char* s = getenv(variable);
  if (!s)
	throw std::runtime_error(std::string("Couldn't read environment variable ") + variable);
#endif
  return s;
}

FILE* Utils::OpenFile(const std::string& fileName, const char* mode)
{
  FILE* file;
#ifdef _WIN32
  if (fopen_s(&file, fileName.c_str(), mode) != 0)
	throw std::runtime_error("Could not open file " + fileName);
#else
  file = fopen(fileName.c_str(), mode);
  if (!file)
	throw std::runtime_error("Could not open file " + fileName);
#endif
  return file;
}
