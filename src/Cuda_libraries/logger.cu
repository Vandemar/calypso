#include "logger.h"

Logger::Logger(std::string fileName) {
  logFile.open(fileName.c_str());
  fileName = fileName;
//  clocks = NULL;
}

//approximate number of clocks to be recorded;
Logger::Logger(std::string fileName, int nClocks) {
  logFile.open(fileName.c_str());
  fileName = fileName;
//  clocks = (Timer*) malloc (sizeof(Timer) * nClocks);
  registeredClocks = 0;
}
  
void Logger::record(std::string comment) {
  logFile << comment << std::endl;
}

/*void Logger::echoAllClocks() {
  this->echoProblemDescription();
  for(int i=0; i<registeredClocks; i++) {
    this->clocks[i].echoTimer(this->getLog()); 
  }
}*/

//ToDo: Create a class that contains the configuation of a run. 
//		i.e, convert the Parameters_s struct to a class
void Logger::recordProblemDescription(int t_lvl, int nShells, int nMeridians) {
  this->t_lvl = t_lvl;
  this->nShells = nShells;
  this->nMeridians = nMeridians;
}

void Logger::echoProblemDescription() {
  logFile << "Truncation Level: " << t_lvl << "\n";
  logFile << "Number of radial shells: " << nShells << "\n";
  logFile << "Number of Meridians: " << nMeridians << "\n"; 
}
 
void Logger::closeStream() {
  logFile.close();
}

/*void Logger::registerTimer(Timer *clock) {
  clocks[registeredClocks] = *clock; 
  registeredClocks++;
}
*/
