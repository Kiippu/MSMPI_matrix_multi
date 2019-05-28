#pragma once
#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <mutex>

const bool s_DEBUG = false;

/// timer logging types/ID's
enum eTimeLogType {
	TT_BEGIN,
	TT_NOW,
	TT_DELTA,
	TT_MULTIPLICATION_BEGIN,
	TT_MAX = UINT16_MAX
} ;

/// typedefs for easy type creation
using TIME_REGISTER = std::map <eTimeLogType, std::chrono::time_point<std::chrono::steady_clock>>;
using TIME_VECTOR_PAIR = std::pair<eTimeLogType, long long>;
using TIME_DISPLAY_VECTOR_PAIR = std::map <eTimeLogType, std::string>;
using TIME_VECTOR = std::vector<TIME_VECTOR_PAIR>;

/*
	Class:			Timer
	Description:	Timer to count elapsed time of tested algorithms
*/

class Timer
{
public:
	/// singleton implementation
	static Timer&getInstance()
	{
		static Timer instance;
		return instance;
	}
	/// no copy
	Timer(Timer&temp) = delete;
	void operator=(Timer const&temp) = delete;
	~Timer() {};

	/// register start timer
	void addStartTime(eTimeLogType eDisplayName, std::string displayName);
	/// registe finish of timer
	void addFinishTime(eTimeLogType id);
	/// print all timers that have finished so far
	void printFinalTimeSheet();
	/// get master delta
	int64_t getDelta();
	/// get master elapsed time 
	int64_t getElapsed();
	/// master update for timers
	bool update();

private:
	Timer();

	int64_t										m_elapsedTime;			/// master time elapsed
	int64_t										m_delta;				/// master delta time
	std::mutex									m_waitMutex;			/// wrapper's mutual exclusion

	std::shared_ptr<TIME_REGISTER>				m_beginTimerList;		/// time lists/maps
	std::shared_ptr<TIME_REGISTER>				m_finishTimerList;
	std::shared_ptr<TIME_VECTOR>				m_finalTimerSheetMs;
	std::shared_ptr<TIME_DISPLAY_VECTOR_PAIR>	m_displayNameList;


};

