#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>

using namespace std;

void query(string line, string name, double *value) {
	if(line.find(name) != string::npos) {
		size_t pos = line.find(":");
		string token = line.substr(pos+1, line.length());
		*value = atof(token.c_str());
	}
}

int main(int argc, char **argv) {
  string fName, ofName;
  if (argc != 2)
    return -1;

  fName = argv[1];
 
  ifstream input;
  input.open(fName.c_str(), ios::in);
  ofstream table;
  table.open("LEG_brokenDown.dat", ios::out);

  string line, token;
  size_t pos;

  double LBT, sp_rlm_h2d, vr_rtm_d2h;
  double LFT, sp_rlm_d2h, vr_rtm_h2d;
  
  table << "t_lvl \t sp_rlm_h2d \t Leg_BWD \t vr_rtm_d2h \t vr_rtm_h2d \t LeG_FWD \t sp_rlm_d2h \n";

  while(getline(input, line)) {
    if(line.find("Truncation level") != string::npos) {
      pos = line.find(":");
      token = line.substr(pos+1, line.length());
      int t_lvl = atoi(token.c_str());
      while(getline(input, line)) {
      	if(line.find("Minimum and maximum") != string::npos)
      		break;
      	else {
            query(line, "cpy sp_rlm host2dev for bwd", &sp_rlm_h2d);
            query(line, "cpy vr_rtm dev2host for bwd", &vr_rtm_d2h);

			query(line, "LGP bwd transform", &LBT); 
			query(line, "LGP fwd transform", &LFT);

			query(line, "cpy vr_rtm host2dev for fwd", &vr_rtm_h2d);
			query(line, "cpy sp_rlm dev2host for fwd", &sp_rlm_d2h);
      	}
      }

	  table << t_lvl << "\t" << sp_rlm_h2d << "\t" << LBT << "\t" << vr_rtm_d2h << "\t" << vr_rtm_h2d << "\t" << LFT << "\t" << sp_rlm_d2h << "\n";
    } 
  }

  input.close();
  table.close();
}
