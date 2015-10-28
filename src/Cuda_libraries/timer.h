#include <mpi.h>
#include <string>
#include <fstream>

class Timer {
    double startTime;
    double endTime;
    double minTime;
    double maxTime;
    double totalTime;
    std::string description;  
    unsigned int counter;

  public:
    Timer();
    Timer(std::string descrition);
    void startTimer();
    void endTimer();
	void echoTimer(std::ofstream *log);
	void echoHeader(std::ofstream *log);
//	Timer& operator=( const Timer &clock);
    std::string whatAmI();
};
