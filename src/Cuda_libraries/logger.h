#include <fstream>
#include <string>

#include "timer.h"

class Logger {
    std::ofstream logFile;
//    Timer *clocks;
    int registeredClocks;
    int t_lvl, nShells, nMeridians;
  public:
    Logger(std::string fileName);
    // nClocks does not have to be exact. 
    Logger(std::string fileName, int nClocks);
    void record(std::string comment);
    std::ofstream* getLog() {return &logFile;}
	void closeStream();
 //   void echoAllClocks();
	void recordProblemDescription(int t_lvl, int nShells, int nMeridians);
	void echoProblemDescription();
//    void registerTimer(Timer *clock);
};
