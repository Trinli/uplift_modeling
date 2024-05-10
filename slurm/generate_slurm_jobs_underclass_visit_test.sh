#!/bin/bash

echo "Generating SLURM jobs for pickles from a given directory.";

if [ $# -eq 0 ]
  then
    echo "No arguments supplied! One argument (directory with pickles) required!";
    exit ;
fi

rm -rf RUN_ALL_SLURM_JOBS.sh

python cvt_underclass_test_gen_slurm_jobs.py $1 cvt baseline
python cvt_underclass_test_gen_slurm_jobs.py $1 cvt noundersampling
python cvt_underclass_test_gen_slurm_jobs.py $1 cvt 1,22,1 #k-search

python cvt_underclass_test_gen_slurm_jobs.py $1 dc baseline
python cvt_underclass_test_gen_slurm_jobs.py $1 dc noundersampling
python cvt_underclass_test_gen_slurm_jobs.py $1 dc 1,22,1 #k-search

python cvt_underclass_test_gen_slurm_jobs.py $1 cvtrf baseline
python cvt_underclass_test_gen_slurm_jobs.py $1 cvtrf noundersampling
python cvt_underclass_test_gen_slurm_jobs.py $1 cvtrf 1,22,1 #k-search

python cvt_underclass_test_gen_slurm_jobs.py $1 dcrf baseline
python cvt_underclass_test_gen_slurm_jobs.py $1 dcrf noundersampling
python cvt_underclass_test_gen_slurm_jobs.py $1 dcrf 1,22,1 #k-search

python cvt_underclass_test_gen_slurm_jobs.py $1 dcsvm baseline
python cvt_underclass_test_gen_slurm_jobs.py $1 dcsvm noundersampling
python cvt_underclass_test_gen_slurm_jobs.py $1 dcsvm 1,22,1 #k-search


