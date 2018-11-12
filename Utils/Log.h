#pragma once

#include <fstream>
#include <sstream>

// This Log class is based on the one described in this article in Dr. Dobb's:
// http://www.drdobbs.com/cpp/logging-in-c/201804215?pgno=1

class LogAdaptor;

class Log
{
  public:
	friend class LogAdaptor;

	enum Levels { None, Error, Warning, Info, Debug1, Debug2, Debug3, Debug4 };

	Log(Levels level) : _level(level) {}
	Log(const Log&) = delete;
	Log& operator=(const Log&) = delete;
	~Log();
	Levels Level() const { return _level;  }
	std::ostringstream& Get();
	const std::ostringstream& Stream() const
	{
	  return _os;
	}

	static void SetReportingLevel(Levels reportingLevel)
	{
	  _reportingLevel = reportingLevel;
	}
	static void SetReportingLevel(std::string);
	static Levels& ReportingLevel() { return _reportingLevel; }
	static void IncludeTime(bool includeTime) { _includeTime = includeTime; }
	static void IncludeThread(bool includeThread) { _includeThread = includeThread; }
	static const char* ToString(Levels level)
	{
	  switch (level)
	  {
		case None: return "None";
		case Error: return "Error";
		case Warning: return "Warning";
		case Info: return "Info";
		case Debug1: return "Debug1";
		case Debug2: return "Debug2";
		case Debug3: return "Debug3";
		case Debug4: return "Debug4";
	  }
	  return "";
	}
	static unsigned ErrorCount() { return _errorCount; }
	static unsigned WarningCount() { return _warningCount; }
  private:
	std::ostringstream _os;
	Levels _level;
	static std::vector<LogAdaptor*> _adaptors;
	static Levels _reportingLevel;
	static unsigned _errorCount;
	static unsigned _warningCount;
	static bool _includeTime;
	static bool _includeThread;
};

class LogAdaptor
{
  public:
	virtual void Save(const Log&) = 0;
	virtual ~LogAdaptor()
	{
	  auto i = std::find(Log::_adaptors.begin(), Log::_adaptors.end(), this);
	  if (i != Log::_adaptors.end())
		Log::_adaptors.erase(i);
	}
  protected:
	LogAdaptor() noexcept
	{
	  Log::_adaptors.push_back(this);
	}
	void Write(std::ostream&, const Log&);

	LogAdaptor(const LogAdaptor&) = delete;
};

class LogConsoleAdaptor : public LogAdaptor
{
  public:
	LogConsoleAdaptor() noexcept {}
	virtual void Save(const Log&);
};

class LogFileAdaptor : public LogAdaptor
{
  public:
	LogFileAdaptor(const std::string& directory, const std::string program);
	virtual ~LogFileAdaptor() override
	{
	  _instance = nullptr;
	}
	virtual void Save(const Log& log) override
	{
	  Write(_fileStream, log);
	  _fileStream.flush();
	}
	static LogFileAdaptor* Instance() { return _instance; }
  private:
	std::ofstream _fileStream;
	std::string _name;
	static LogFileAdaptor* _instance;
};

class LogTestAdaptor : LogAdaptor
{
  public:
	LogTestAdaptor(Log::Levels level = Log::Info) noexcept
	{
	  Log::SetReportingLevel(level);
	  Log::IncludeTime(false);
	  Log::IncludeThread(false);
	}
	virtual void Save(const Log& log) override
	{
	  _messages.push_back(log.Stream().str());
	}
	const std::vector<std::string>& Messages() const { return _messages; }
	void Clear()
	{
	  _messages.clear();
	}
  private:
	std::vector<std::string> _messages;
};

inline std::ostream& operator<<(std::ostream& os, const std::vector<char*>& v)
{
  for (auto s : v)
#ifdef _WIN32
	os << s << "\r\n";
#else
	os << s << '\n';
#endif
  return os;
}

// This uses if/else with an empty "if" branch so that it doesn't do unexpected
// things if the LOG macro is used inside an "if" statement without curly braces.

#define LOG(level) \
if (Log::level > Log::ReportingLevel()) {} else Log(Log::level).Get()
