#define _CRT_SECURE_NO_WARNINGS
#include "Timer.h"
#include <ctime>

Timer::Timer()
{
	m_beginTimerList = std::make_shared<TIME_REGISTER>();
	m_finishTimerList = std::make_shared<TIME_REGISTER>();
	m_finalTimerSheetMs = std::make_shared<TIME_VECTOR>();
	m_displayNameList = std::make_shared<TIME_DISPLAY_VECTOR_PAIR>();
	m_elapsedTime = 0;

	addStartTime(eTimeLogType::TT_BEGIN, "Total program runtime");
	addStartTime(eTimeLogType::TT_DELTA, "Time since last update was competed");
}

// log start timer
void Timer::addStartTime(eTimeLogType eDisplayName, std::string displayName)
{
	std::unique_lock<std::mutex> scopedLock(m_waitMutex);
	m_displayNameList->insert(std::make_pair(eDisplayName, displayName));
	m_beginTimerList->insert(std::make_pair(eDisplayName, std::chrono::high_resolution_clock::now()));
	if ((int)eDisplayName > (int)eTimeLogType::TT_MAX) {
		printf("Timer enum over MAX limit: %d !\n", eDisplayName);
		printf("(int)eTimeLogType::TT_MAX: %d !\n", (int)eTimeLogType::TT_MAX);
	}
};
// log finish timer
void Timer::addFinishTime(eTimeLogType id)
{
	//log finish timer
	std::unique_lock<std::mutex> scopedLock(m_waitMutex);
	auto finishTimer = std::chrono::high_resolution_clock::now();
	m_finishTimerList->insert(std::make_pair(id, finishTimer));
};
// print time sheet of finihsed timers
void Timer::printFinalTimeSheet()
{
	// iterate through all values in timesheet and print them.
	std::unique_lock<std::mutex> scopedLock(m_waitMutex);
	for (auto & obj : *m_finishTimerList)
	{
		auto differenceInTime = m_finishTimerList->at(obj.first) - m_beginTimerList->at(obj.first);
		long long time = std::chrono::duration_cast<std::chrono::milliseconds>(differenceInTime).count();
		std::cout << m_displayNameList->at(obj.first) << " : " << std::to_string(time) << "ms / " << std::to_string(float(float(time) / 1000.f)) << "sec to execute\n" << std::endl;
	}
}

int64_t Timer::getDelta()
{
	std::unique_lock<std::mutex> scopedLock(m_waitMutex);
	return m_delta;
}

int64_t Timer::getElapsed()
{
	std::unique_lock<std::mutex> scopedLock(m_waitMutex);
	return m_elapsedTime;
}

bool Timer::update()
{
	std::unique_lock<std::mutex> scopedLock(m_waitMutex);
	auto now = std::chrono::high_resolution_clock::now();
	// set elapsed time
	auto differenceInTime = now - m_beginTimerList->at(eTimeLogType::TT_BEGIN);
	m_elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(differenceInTime).count();

	//set delta
	differenceInTime = now - m_beginTimerList->at(eTimeLogType::TT_DELTA);
	m_delta = std::chrono::duration_cast<std::chrono::milliseconds>(differenceInTime).count();
	m_beginTimerList->at(eTimeLogType::TT_DELTA) = now;

	return true;
}
