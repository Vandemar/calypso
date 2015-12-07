set terminal jpeg
set output "SHT_BrokenDown.jpeg"
set title "Components of Spherical Harmonic Transform"
set key outside box
set xlabel "Truncation Level"
set ylabel "Time (s)"
set logscale x
set logscale y
set xrange [20:200]

plot "SHT_brokenDown.dat" using 1:2 title "SHT_BWD" w linespoints, \
"SHT_brokenDown.dat" using 1:3 title "transfer_rj_2_rlm" w linespoints, \
"SHT_brokenDown.dat" using 1:4 title "Legendre_BWD" w linespoints, \
"SHT_brokenDown.dat" using 1:5 title "transfer_rtm_rtp" w linespoints, \
"SHT_brokenDown.dat" using 1:6 title "FFT_4_BWD" w linespoints, \
"SHT_brokenDown.dat" using 1:7 title "SHT_FWD" w linespoints, \
"SHT_brokenDown.dat" using 1:8 title "FFT_4_FWD" w linespoints, \
"SHT_brokenDown.dat" using 1:9 title "transfer_rtp_rtm" w linespoints, \
"SHT_brokenDown.dat" using 1:10 title "Legendre_FWD" w linespoints, \
"SHT_brokenDown.dat" using 1:11 title "transfer_rlm_rj" w linespoints, \
"SHT_brokenDown.dat" using 1:12 title "coriolis_term" w linespoints
