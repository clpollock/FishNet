#include "stdafx.h"
#include "Tokenizer.h"
#include "StringUtils.h"

std::string Tokenizer::NextToken()
{
  SkipWhitespace();
  const char* start = _position;
  while (_position < _end && !StringUtils::IsWhiteSpace(*_position))
	++_position;
  std::string s(start, _position - start);
  return s;
}

void Tokenizer::SkipWhitespace()
{
  while (_position < _end && StringUtils::IsWhiteSpace(*_position))
    ++_position;
}

void Tokenizer::Trim()
{
  SkipWhitespace();
  while (_end > _position && StringUtils::IsWhiteSpace(*(_end - 1)))
    --_end;
}
