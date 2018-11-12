#include "stdafx.h"
#include "StringUtils.h"

#ifdef _WIN32
std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> StringUtils::_converter;
#endif

bool StringUtils::ToBool(std::string v)
{
  ToLower(v);
  if (v == "y" || v == "t" || v == "yes" || v == "true" || v == "1")
	return true;
  if (v == "n" || v == "f" || v == "no" || v == "false" || v == "0")
	return false;
  throw std::runtime_error("Invalid boolean value: " + v);
}

void StringUtils::SplitCSV(std::vector<std::string>& fields, std::string& line)
{
  // We need to manually remove carriage return characters if we're reading a Windows file under Linux.
  if (!line.empty() && line.back() == '\r')
	line.pop_back();
  fields.clear();
  std::string field;
  field.reserve(line.length());
  bool quoted = false;
  const char* s = line.c_str();
  char c = *(s++);
  while (c)
  {
	if (c == '"')
	{
	  if (*s == '"')
	  {
		field += '"';
		++s;
	  }
	  else
	  {
		quoted = !quoted;
	  }
	}
	else if (c == ',' && !quoted)
	{
	  while (!field.empty() && IsWhiteSpace(field.back()))
		field.pop_back();
	  fields.emplace_back(field);
	  field.clear();
	}
	else if (!IsWhiteSpace(c) || !field.empty()) // Don't add leading whitespace.
	{
	  field += c;
	}
	c = *(s++);
  }
  if (quoted)
	throw std::runtime_error("Unbalanced quotation marks.");
  while (!field.empty() && IsWhiteSpace(field.back()))
	field.pop_back();
  fields.emplace_back(field);
}

bool StringUtils::IsNumeric(const std::string& s)
{
  if (s.empty())
	return false;
  for (auto c : s)
	if (c < '0' || c > '9')
	  return false;
  return true;
}
