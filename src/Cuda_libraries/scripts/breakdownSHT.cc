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
  table.open("SHT_brokenDown.dat", ios::out);

  string line, token;
  size_t pos;

  double SHT_b, transfer_rj_2_rlm, coriolis_term, transfer_rtm_2_rtp, FFT_b;
  double LBT;
  double SHT_f, transfer_rtp_2_rtm, transfer_rlm_2_rj, FFT_f;
  double LFT;
  
  table << "t_lvl \t SHT_BWD \t transfer_rj_2_rlm \t Leg_bwd \t transfer_rtm_2_rtp \t FFT_b \t SHT_FWD \t FFT_f \t transfer_rtp_2_rtm \t LEG_FWD \t transfer_rlm_2_rj \t coriolis_term \n";

  while(getline(input, line)) {
    if(line.find("Truncation level") != string::npos) {
      pos = line.find(":");
      token = line.substr(pos+1, line.length());
      int t_lvl = atoi(token.c_str());
      while(getline(input, line)) {
      	if(line.find("Minimum and maximum") != string::npos)
      		break;
      	else {
      		query(line, "sph backward transform", &SHT_b);
      		query(line, "transfer rj  => rlm", &transfer_rj_2_rlm);
      		query(line, "Legendre backward transform", &LBT);
      		query(line, "transfer rtm => rtp", &transfer_rtm_2_rtp);
            query(line, "Fourier transform bwd", &FFT_b);

      		query(line, "sph forward transform", &SHT_f);
            query(line, "Fourier transform fwd", &FFT_f);
      		query(line, "transfer rtp => rtm", &transfer_rtp_2_rtm);
      		query(line, "Legendre forward transform", &LFT);
      		query(line, "transfer rlm => rj", &transfer_rlm_2_rj);	

      		query(line, "Coriolis term", &coriolis_term);
      	}
      }
	   table << t_lvl << "\t" << SHT_b << "\t" << transfer_rj_2_rlm << "\t" << LBT << "\t" << transfer_rtm_2_rtp << "\t" << FFT_b << "\t" 
	     << SHT_f << "\t" << FFT_f << "\t" << transfer_rtp_2_rtm << "\t" << LFT << "\t" << transfer_rlm_2_rj << "\t" << coriolis_term << "\t"
	       << endl; 
    } 
  }

  input.close();
  table.close();
}
