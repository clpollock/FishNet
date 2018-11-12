#pragma once

#ifdef _WIN32
#include <codecvt>
const char PATH_SEPARATOR = '\\';
#else
const char PATH_SEPARATOR = '/';
#endif

class StringUtils
{
  public:
	static inline void ToLower(std::string& s)
	{
	  transform(s.begin(), s.end(), s.begin(), ::tolower);
	}
	static inline void ToLower(char* s)
	{
	  while (*s)
	  {
		*s = ::tolower(*s);
		++s;
	  }
	}
	static inline void ToUpper(std::string& s)
	{
	  transform(s.begin(), s.end(), s.begin(), ::toupper);
	}
	static bool ToBool(std::string);
  	static void SplitCSV(std::vector<std::string>& fields, std::string& line);
	static bool IsWhiteSpace(char c)
	{
	  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
	}
	static bool IsNumeric(const std::string&);
#ifdef _WIN32
	static inline std::wstring ConvertToUTF16(const char* s)
	{
	  return _converter.from_bytes(s);
	}
	static inline std::wstring ConvertToUTF16(const std::string& s)
	{
	  return _converter.from_bytes(s);
	}
	static inline std::string ConvertToUTF8(const wchar_t* s)
	{
	  return _converter.to_bytes(s);
	}
	static inline std::string ConvertToUTF8(const std::wstring& s)
	{
	  return _converter.to_bytes(s);
	}
  private:
	static std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> _converter;
#endif
};
