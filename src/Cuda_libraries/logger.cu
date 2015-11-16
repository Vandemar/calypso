#include "logger.h"

Logger::Logger(std::string fileName) {
  logFile.open(fileName.c_str(), std::fstream::app);
  fileName = fileName;
  clocks = NULL;
  registeredClocks = 0;
}

//approximate number of clocks to be recorded;
Logger::Logger(std::string fileName, int nClocks) {
  logFile.open(fileName.c_str(), std::fstream::app);
  fileName = fileName;
  clocks = new Timer*[nClocks];
  registeredClocks = 0;
}
 
Logger::~Logger() {
  //delete[] clocks;
} 

void Logger::record(std::string comment) {
  logFile << comment << std::endl;
}

void Logger::echoAllClocks() {
  this->echoProblemDescription();
  this->clocks[0]->echoHeader(this->getLog());
  for(int i=0; i<registeredClocks; i++) {
    this->clocks[i]->echoTimer(this->getLog()); 
  }
  logFile << std::endl;
}

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

void Logger::registerTimer(Timer *clock) {
  if(!clock->getRegistrationStatus()) {
    clock->setRegistrationStatus();
    clocks[registeredClocks] = clock; 
    registeredClocks++;
  }
}

