"""
Script for generating slurm scripts for extended undersampling experiments.
Scripts will go into ./tmp/

This file will generate scripts for the uplift_rf and dc-lr experiments
with __split undersampling__.
"""

import os
from pathlib import Path



def generate_slurm_scripts(clear_old_scripts=False):
    """
    Function for generating slurm scripts for uplift-rf grid search experiment.

    1. Check whether ./tmp/ exists. Create if not.
    2. Remove *.job from ./tmp/
    3. Create a bunch of slurm_{}.job's into ./tmp/
    4. Done. The actual jobs are submitted by sbatch ./tmp/*.job

    Misc:
    -Perhaps check how long it takes to do a job with k_t or k_c = 1
     and run them separately? Will need more time at least.
    """
    # Not exactly binary progression.
    # Also, jobs with larger k_* should be clearly faster.
    # Big question is whether a job with big k_t/k_c and k_c/k_t
    # == 1 would be fast. Just try it. At least with k_c 16 and
    # the data with visit label, this was the case.
    # Check if tmp exists:
    if not os.path.exists('./tmp/'):
        # Create path:
        os.mkdir('./tmp/')
    # Remove *.job's from tmp
    # os.remove('./tmp/*.job')
    if clear_old_scripts:
        for p in Path("./tmp/").glob("*.job"):
            p.unlink()
    # Empty list for files:
    files = []
    # Create new jobs:
    # Initial results not convincing. Maybe create more dense search grid?
    # Also add combinations for special cases (k-undersampling included as
    # k_c and k_t vectors are identical). 
    # Specia case p_t == p_c
    #k_t = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    #k_c = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # New scripts with p_t* == p_c*
    #r = 1.548946888639783  # this r is for the Criteo-dataset (criteo1?)
    k_t = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    #k_c = [r * item for item in k_t] + k_t
    k_c = k_t
    #k_t = [1] + [i * 10 for i in range(25)]
    #k_c = k_t
    # datasets = ['Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv',
    # 'starbucks.csv', 'voter/voter_1_to_1_.5_dataset.csv']
    datasets = ['zenodo_modified.csv']
    parameters = []
    models = ['uplift_dc', 'uplift_rf']
    for dataset in datasets:
        for model in models:
            for i in k_t:
                for j in k_c:
                    # Set "mem" depending on what values k_t and k_c has.
                    # Might get resources faster.
                    time = '23:59:59'
                    mem = '64G'
                    part = "medium"
                    if True:
                        # Skip all of conditions below.
                        pass
                    elif i == 1 and j == 1:
                        time = '23:59:59'
                        mem = '64G'
                        part = "medium"
                    elif i == 1 or j == 1:
                        time = '23:59:59'
                        mem = '32G'
                        part = "medium"
                    elif i > 4 and j > 4:
                        time = '1:00:00'
                        mem = '8G'
                        part = "short"
                    elif i > 1 and j > 1:
                        # Basically "else", or 4 >= k_* > 1
                        # Should not enter if the previous condition
                        # was satisfied.
                        time = '8:00:00'
                        mem = '16G'
                        part = 'short'
                    # For every i, j combination, formulate text
                    # and write to separate slurrm file.
                    if True:
                        # Skip conditions below
                        pass
                    elif model == 'uplift_dc':
                        # Change everything to basic settings.
                        time = '1:00:00'
                        mem = '8G'
                        part = 'short'
                    text = """#!/bin/bash
#SBATCH --job-name={0}_with_kt_{1}_and_kc_{2}
#SBATCH -o ./results_{0}/grid_search/result_{0}_with_kt_{1}_and_kc_{2}.txt
#SBATCH -M ukko
#SBATCH -c 1
#SBATCH -t {3}
#SBATCH --mem={4}
#SBATCH -p{5}
export PYTHONPATH=$PYTHONPATH:.
srun hostname
srun sleep 5
srun python -m experiments.run_crf_experiment {6} ./datasets/ 1.0 {0} {1} {2}
""".format(model, i, j, time, mem, part, dataset)
                    # Alternative for voter-experiments: srun python -m experiments.run_crf_experiment voter_1_to_1_.5_dataset.csv ./datasets/voter/ 1.0 {0} {1} {2}
                    # Alternative row for criteo-experiments: srun python -m experiments.run_crf_experiment criteo1_1to1.csv ./datasets/criteo1/ 1.0 {0} {1} {2}
                    parameters.append("{0} {1} {2}".format(model, i, j))
                    # Write text to file:
                    #if k_t.index(i) == k_c.index(j): # Skipped diagonal at some point.
                    if dataset == 'voter/voter_1_to_1_.5_dataset.csv':
                        dataset_name = 'voter'
                    elif dataset == 'Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv':
                        dataset_name = 'hillstrom'
                    elif dataset == 'starbucks.csv':
                        dataset_name = 'starbucks'
                    elif dataset == 'zenodo_modified.csv':
                        dataset_name = 'zenodo'
                    tmp_filename = './tmp/slurm_{}_{}_kt{}_kc{}.job'.format(dataset_name, model, i, j)
                    with open(tmp_filename, 'w') as handle:
                        handle.write(text)
                    files.append('./tmp/slurm_{}_{}_kt{}_kc{}.job'.format(dataset_name, model, i, j))
                    # Last row is "num uplift_rf k"
                    # Also change the output file and perhaps even dir.
                    print(text)
    with open('./tmp/bash_script.sh', 'a') as handle:
        handle.write('#!/bin/bash\n')
        for item in files:
            handle.write("sbatch " + item + "\n")
    with open('./tmp/parameters.txt', 'w') as handle:  # How could this even work when half of the info is missing?!? Use bash-script.
        for item in parameters:
            handle.write(item + '\n')

