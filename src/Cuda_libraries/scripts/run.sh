#!/bin/bash

./startRuns control_MHD_template control_sph_shell_template 31
../../build/calypso/STATIC/release/bin/gen_sph_grids
../../build/calypso/STATIC/release/bin/sph_mhd
./startRuns control_MHD_template control_sph_shell_template 63
../../build/calypso/STATIC/release/bin/gen_sph_grids
../../build/calypso/STATIC/release/bin/sph_mhd
./startRuns control_MHD_template control_sph_shell_template 83
../../build/calypso/STATIC/release/bin/gen_sph_grids
../../build/calypso/STATIC/release/bin/sph_mhd
./startRuns control_MHD_template control_sph_shell_template 103
../../build/calypso/STATIC/release/bin/gen_sph_grids
../../build/calypso/STATIC/release/bin/sph_mhd
./startRuns control_MHD_template control_sph_shell_template 127
../../build/calypso/STATIC/release/bin/gen_sph_grids
../../build/calypso/STATIC/release/bin/sph_mhd

