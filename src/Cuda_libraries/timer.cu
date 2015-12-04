#include <limits>
#include <iomanip>
 
#include "timer.h"

Timer::Timer() {
  startTime = 0;
  endTime = 0;
  minTime = std::numeric_limits<double>::max();
  maxTime = std::numeric_limits<double>::min();
  totalTime = 0;
  counter = 0;
  registerStatus = false;
}

Timer::Timer(std::string whatAmI) {
  startTime = 0;
  endTime = 0;
  minTime = std::numeric_limits<double>::max();
  maxTime = std::numeric_limits<double>::min();
  totalTime = 0;
  counter = 0;
  description.assign(whatAmI);
  registerStatus = false;
}
 
void Timer::startTimer() {
  counter++;
  startTime = MPI_Wtime();
}

void Timer::endTimer() {
  endTime = MPI_Wtime();
  double diff = endTime - startTime;
  totalTime += diff;
  if ( diff < minTime) minTime = diff;
  if ( diff > maxTime) maxTime = diff; 
}

void Timer::resetTimer() {
  startTime = 0;
  endTime = 0;
  minTime = std::numeric_limits<double>::max();
  maxTime = std::numeric_limits<double>::min();
  totalTime = 0;
}

void Timer::echoHeader(std::ofstream *log) {
  *log << std::setw(25) << " " << "\t" << "Minimum Time (s)" << "\tMaximum Time (s)" << "\tAverage Time (s)" << std::endl; 
}
 
void Timer::echoTimer(std::ofstream *log) {
  *log << std::setw(25) << this->whatAmI() << "\t" << minTime << "\t" << maxTime << "\t" << totalTime/counter << std::endl;  
}

std::string Timer::whatAmI() {
  return description;
} 

void Timer::setWhatAmI(std::string context) {
  description.assign(context);  
} 

Timer& Timer::operator=( const Timer &clock) {
  this->startTime = clock.startTime;
  this->endTime = clock.endTime;
  this->minTime = clock.minTime;
  this->maxTime = clock.maxTime;
  this->totalTime = clock.totalTime;
  this->counter = clock.counter;
  this->registerStatus = clock.registerStatus;
  this->description.assign(clock.description);
  return *this;
}

