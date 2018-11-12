#include "stdafx.h"
#include "Log.h"
#include "StringUtils.h"
#include <iomanip>
#include <thread>

std::vector<LogAdaptor*> Log::_adaptors;
Log::Levels Log::_reportingLevel = Log::Debug4;
unsigned Log::_errorCount = 0;
unsigned Log::_warningCount = 0;
bool Log::_includeTime = true;
bool Log::_includeThread = false;

Log::~Log()
{
  for (LogAdaptor* adaptor : _adaptors)
	adaptor->Save(*this);
}

std::ostringstream& Log::Get()
{
  if (_includeTime)
  {
#ifdef _WIN32
	SYSTEMTIME time;
	GetLocalTime(&time);
	_os << time.wYear << '/'
	  << std::setw(2) << std::setfill('0') << time.wMonth << '/'
	  << std::setw(2) << std::setfill('0') << time.wDay << ' '
	  << std::setw(2) << std::setfill('0') << time.wHour << ':'
	  << std::setw(2) << std::setfill('0') << time.wMinute << ':'
	  << std::setw(2) << std::setfill('0') << time.wSecond << '.'
	  << std::setw(3) << std::setfill('0') << std::left << time.wMilliseconds << ' ';
#else
	struct timeval t;
	struct tm* time;
	gettimeofday(&t, nullptr);
	time = localtime(&t.tv_sec);
	_os << time->tm_year + 1900 << '/'
	  << std::setw(2) << std::setfill('0') << time->tm_mon + 1 << '/'
	  << std::setw(2) << std::setfill('0') << time->tm_mday << ' '
	  << std::setw(2) << std::setfill('0') << time->tm_hour << ':'
	  << std::setw(2) << std::setfill('0') << time->tm_min << ':'
	  << std::setw(2) << std::setfill('0') << time->tm_sec << '.'
	  << std::setw(3) << std::setfill('0') << std::left << t.tv_usec << ' ';
#endif
  }
  if (_includeThread)
	_os << "thread " << std::this_thread::get_id() << ' ';
  _os << ToString(_level) << ":\t";
  if (_level == Error)
	++_errorCount;
  else if (_level == Warning)
	++_warningCount;
  return _os;
}

void Log::SetReportingLevel(std::string s)
{
  StringUtils::ToLower(s);
  if (s == "none")
	_reportingLevel = None;
  else if (s == "error")
	_reportingLevel = Error;
  else if (s == "warning")
	_reportingLevel = Warning;
  else if (s == "info")
	_reportingLevel = Info;
  else if (s == "debug1")
	_reportingLevel = Debug1;
  else if (s == "debug2")
	_reportingLevel = Debug2;
  else if (s == "debug3")
	_reportingLevel = Debug3;
  else if (s == "debug4")
	_reportingLevel = Debug4;
  else
	throw std::runtime_error("Invalid log reporting level: " + s + " Valid levels are Error, Warning, Info, Debug1, Debug2, Debug3, and Debug4.");
}

void LogAdaptor::Write(std::ostream& stream, const Log& log)
{
  std::string s = log.Stream().str();
  if (!s.empty())
  {
#ifdef _WIN32
	if (s.back() != '\n')
	{
	  s.append("\r\n");
	}
	else if (s.length() > 1 && s[s.length() - 2] != '\r')
	{
	  s[s.length() - 1] = '\r';
	  s.append("\n");
	}
#else
	if (s.back() != '\n')
	  s += '\n';
#endif
	stream << s;
  }
}

void LogConsoleAdaptor::Save(const Log& log)
{
  Write(log.Level() <= Log::Warning ? std::cerr : std::cout, log);
}

LogFileAdaptor* LogFileAdaptor::_instance = nullptr;

LogFileAdaptor::LogFileAdaptor(const std::string& directory, const std::string program)
{
  if (_instance)
	throw std::runtime_error("LogFileAdaptor already created.");
  _instance = this;
  char logPath[256];
#ifdef _WIN32
  SYSTEMTIME time;
  GetLocalTime(&time);
  sprintf_s(logPath, 256, "%s\\%s_%d%02d%02d-%02d%02d%02d.log", directory.c_str(), program.c_str(),
	time.wYear, time.wMonth, time.wDay, time.wHour, time.wMinute, time.wSecond);
  _fileStream.open(StringUtils::ConvertToUTF16(logPath), std::ofstream::out | std::ofstream::binary);
#else
  time_t t = ::time(0);
  tm* time = localtime(&t);
  snprintf(logPath, 256, "%s/%s_%d%02d%02d-%02d%02d%02d.log", directory.c_str(), program.c_str(),
	time->tm_year + 1900, time->tm_mon + 1, time->tm_mday, time->tm_hour, time->tm_min, time->tm_sec);
  _fileStream.open(logPath, std::ofstream::out | std::ofstream::binary);
#endif
  _name = logPath;
}
