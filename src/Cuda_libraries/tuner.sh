#!/bin/bash

# The different tuning parameters are defined below:
NTHETA=192
N_CONFIGS=12
THREADS_PER_BLOCK=(1 2 3 4 8 12 16 24 48 64 91 192)
CUB_THETA_PER_THREAD=(192 91 64 48 24 16 12 8 4 3 2 1)
CUB_NVECTOR=4
CUB_NCOMPS=13
SRC_DIR="/home/cig/workspace/calypso/"
RUN_DIR="/home/cig/workspace/runs/calypso/dynamobench_case_0"
BUILD_DIR="/home/cig/workspace/runs/calypso/dynamobench_case_0/build"

CPU_COUNT=6
# The set of rules that must be adhered

# The build steps 

if ! [ -d $BUILD_DIR ]; then
  mkdir $BUILD_DIR
fi

for index in `seq $N_CONFIGS`; do 
  cd $BUILD_DIR

  echo ${THREADS_PER_BLOCK[$((index-1))]}
  /home/cig/Downloads/cmake-2.8.12-rc1-Linux-i386/bin/cmake -DCMAKE_BUILD_TYPE=RELEASE -DCUB=ON -DCUDA_NVCC_FLAGS="-DCUB_THREADS_PER_BLOCK=${THREADS_PER_BLOCK[$((index-1))]} -DCUB_NVECTOR=$CUB_NVECTOR -DCUB_NCOMPS=$CUB_NCOMPS -DCUB_THETA_PER_THREAD=${CUB_THETA_PER_THREAD[$((index-1))]}" $SRC_DIR &> /dev/null
  make -j $CPU_COUNT &> /dev/null

  cd $RUN_DIR 
  
  mpirun -np 4 $BUILD_DIR/bin/sph_mhd
done


# If succesfully built, then test, collect data and catalog state of each configuration.

# Build best parameterized build.
