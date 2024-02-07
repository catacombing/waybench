#!/usr/bin/env sh

# Plot benchmark results as SVG.

# Make sure gnuplot is installed.
if ! [ -x "$(command -v gnuplot)" ]; then
    printf "command not found: gnuplot\n"
    exit 1
fi

# Ensure at least one input and output file is present.
if [ $# -lt 1 ]; then
    printf "Usage: gnuplot.sh <RESULTS.CSV>\n"
    exit 1
fi

input_file="$1"
output_file="./results.svg"

gnuplot_script="\
    set terminal svg noenhanced size 1000,750 background rgb 'white'
    set output \"${output_file}\"
    set xlabel \"sample\"
    set ylabel \"render time (ms)\"
    set datafile separator ','
    set key autotitle columnhead
    plot for [i=1:*] \"${input_file}\" using 0:i with linespoint"
printf "${gnuplot_script}" | gnuplot
