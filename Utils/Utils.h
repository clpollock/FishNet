#pragma once

class Utils
{
  public:
	static std::string GetEnv(const char* variable);
	static std::string GetEnv(const std::string& variable)
	{
	  return GetEnv(variable.c_str());
	}
	static FILE* OpenFile(const std::string& fileName, const char* mode);
};
