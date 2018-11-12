#pragma once

#include <string>
#include <string.h>

class Tokenizer
{
  public:
	// Tokenizer doesn't make a copy of the string so the string data must not be
	// deleted or allowed to go out of scope while the Tokenizer is in use.
	// The string does not have to be null terminated.
	Tokenizer(const char* s, size_t length)
	  : _position(s), _end(s + length) {}
	// This version of Tokenizer requires the string to be null terminated.
	Tokenizer(const char* s)
	  : _position(s), _end(s + strlen(s)) {}
	Tokenizer(const std::string& s) 
	  : _position(s.c_str()), _end(_position + s.length()) {}
	std::string NextToken();
	std::string Remaining() const
	{
	  return std::string(_position, _end - _position);
	}
	const char* Position() const { return _position; }
	bool IsEmpty() const
	{
	  return _position == _end;
	}
	void SkipWhitespace();
  private:
	void Trim();
	// Current position in the string.
	const char* _position;
	// End is the address immediately after the last character in the string.
	const char* _end;
};
