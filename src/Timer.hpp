/*********************************
Developer: Maksudul Alam
Oak Ridge National Laboratory
*********************************/

#ifndef TIMER_H_
#define TIMER_H_

#ifdef WIN32   // Windows system specific
#include <windows.h>
#else          // Unix based system specific
#include <sys/time.h>
#endif

class Timer
{
public:
    Timer();
    ~Timer();

    void start();
    void stop();
    void reset();

    double getElapsedTime();
    double getElapsedTimeInSec();
    double getElapsedTimeInMilliSec();
    long getElapsedTimeInMicroSec();

protected:
    long getLastElapsed();

private:
    double startTimeInMicroSec;
    double endTimeInMicroSec;

    long elapsedTimeInMicroSec;

    int stopped;
#ifdef WIN32
    LARGE_INTEGER frequency;
    LARGE_INTEGER startCount;
    LARGE_INTEGER endCount;
#else
    timeval startCount;
    timeval endCount;
#endif
};

#endif /* TIMER_H_ */
