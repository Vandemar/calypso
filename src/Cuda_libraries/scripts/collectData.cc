#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>

using namespace std;

int nTheta(int l) {
  return ((l+1) + (l+1)/2);
}

int nPhi(int theta) {
  return theta*2;
}

//Arguments must have a different filename than control_MHD and control_sph_shell
int main(int argc, char **argv) {
  if(argc != 4) 
    return -1;
 
  string ctl_MHD = argv[1]; 
  string ctl_SPH = argv[2]; 
  int t_lvl = stoi(argv[3]); 

  string line; 
  string fNameMHD = "control_MHD";
  string fNameSPH = "control_sph_shell";
  string tmp;

  string dir  = "sph_lm" + to_string((long long int) t_lvl) + "r63c_1";
  mkdir(dir.c_str(), 0755);

  ifstream MHD, SPH;
  ofstream MHD_mod, SPH_mod;

  MHD.open(ctl_MHD.c_str(), ios::in);
  SPH.open(ctl_SPH.c_str(), ios::in); 

  MHD_mod.open(fNameMHD, ios::out); 
  SPH_mod.open(fNameSPH, ios::out);
  while( getline(MHD, line)) {
    //control MHd 
    if(line.find("sph_file_prefix") != string::npos && line.find("in") != string::npos) {
      tmp = "'sph_lm" + to_string((long long int) t_lvl) + "r63c_1/in'";
      MHD_mod << "    sph_file_prefix             " << tmp << endl;
    }
    else
      MHD_mod << line << endl;
  }

  while( getline(SPH, line)) {
    //control_sph_shell
    if(line.find("sph_file_prefix") != string::npos) {
      tmp = "'sph_lm" + to_string((long long int) t_lvl) + "r63c_1/in'";
      SPH_mod << "    sph_file_prefix             " << tmp << endl;
    }
    else if(line.find("truncation_level_ctl") != string::npos) {
      tmp = to_string((long long int) t_lvl);
      SPH_mod << "      truncation_level_ctl      " << tmp << endl; 
    }
    else if(line.find("ngrid_meridonal_ctl") != string::npos) {
      tmp = to_string((long long int) nTheta(t_lvl));
      SPH_mod << "    ngrid_meridonal_ctl      " << tmp << endl;
    }
    else if(line.find("ngrid_zonal_ctl") != string::npos) {
      tmp = to_string((long long int) nPhi(nTheta(t_lvl)));
      SPH_mod << "    ngrid_zonal_ctl          " << tmp << endl;
    }
    else
      SPH_mod << line << endl;
  }

  MHD_mod.close();
  SPH_mod.close();
  MHD.close();
  SPH.close();
  
  return 1;
} 
