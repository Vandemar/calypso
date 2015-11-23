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
    bool registerStatus;
  public:
    Timer();
    Timer(std::string whatAmI);
    void startTimer();
    void endTimer();
    double elapsedTime() {return totalTime;}
    void resetTimer();
	void echoTimer(std::ofstream *log);
	void echoHeader(std::ofstream *log);
    bool getRegistrationStatus() {return registerStatus;}
    void setRegistrationStatus() {registerStatus=true;}
	Timer& operator=( const Timer &clock);
    std::string whatAmI();
	void setWhatAmI(std::string context);
};
