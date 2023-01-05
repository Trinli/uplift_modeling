"""
Script for generating slurm scripts for extended undersampling experiments.
Scripts will go into ./tmp/

This contains everything except for split undersampling experiments.
I.e. baselines, k-undersampling, and naive (+dc-undersampling) with
all models and none, div-by-k, and isotonic regression for calibration.
"""
import os
from pathlib import Path


def generate_slurm_scripts_dc_cvt(clear_old_scripts=False):
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

    #datasets = ['hillstrom', 'starbucks', 'voter']  #['zenodo']  #['voter']  #['criteo1', 'criteo2']  # , 'hillstrom']
    # datasets = ['test']
    datasets = ['zenodo']

    configs = [  # Note that uplift-rf and uplift neural net will need separate k-selection.
        # Baselines:
        ['none', 'dc_lr', 'none', './results/baselines.csv'],
        ['none', 'cvt_lr', 'none', './results/baselines.csv'],
        # k-undersampling with 
        ['k_undersampling', 'dc_lr', 'div_by_k', './results/k_undersampling_dc_lr_div_by_k.csv'],
        ['k_undersampling', 'cvt_lr', 'div_by_k', './results/k_undersampling_cvt_lr_div_by_k.csv'],
        # k-undersampling with calibration
        ['k_undersampling', 'dc_lr', 'isotonic', './results/k_undersampling_dc_lr_isotonic.csv'],
        ['k_undersampling', 'cvt_lr', 'isotonic', './results/k_undersampling_cvt_lr_isotonic.csv'],
        # Naive
        ['naive_undersampling', 'dc_lr', 'isotonic', './results/naive_undersampling_dc_lr_isotonic.csv'],
        ['naive_undersampling', 'cvt_lr', 'isotonic', './results/naive_undersampling_cvt_lr_isotonic.csv'],
        # dc-undersampling + calibration (a second baseline)
        ['dc_undersampling', 'naive_dc', 'none', './results/dc_undersampling_naive_dc_none.csv']
    ]

    for dataset in datasets:
        for item in configs:
            # Model is DC-LR or CVT-LR
            time = "8:00:00"
            mem = "16G"
            part = "short"
            # dataset defined by loop
            undersampling = item[0]
            model = item[1]
            correction = item[2]
            output_file = item[3]
            text = """#!/bin/bash
#SBATCH --job-name={0}_{1}_{2}_{3}
#SBATCH -o ./slurm_out/result_{0}_{1}_{2}_{3}.txt
#SBATCH -M ukko
#SBATCH -c 1
#SBATCH -t {4}
#SBATCH --mem={5}
#SBATCH -p{6}
export PYTHONPATH=$PYTHONPATH:.
srun hostname
srun sleep 5
srun python -m experiments.undersampling_experiments {0} {1} {2} {3} {7}
""".format(dataset, undersampling, model, correction, time, mem, part, output_file)

            # Write text to file:
            tmp_filename = './tmp/slurm_{0}_{1}_{2}_{3}.job'.format(
                dataset, undersampling, model, correction,
                time, mem, part, output_file)
            with open(tmp_filename, 'w') as handle:
                handle.write(text)
            # Add filename to list to later write to bash-script.
            files.append(tmp_filename)
            # Also change the output file and perhaps even dir.
            print(text)
    # Add all to a bash-script to submit slurm-jobs:    
    # # Keep track of files in list and write to bash-script:
    with open('./tmp/bash_script.sh', 'w') as handle:
        handle.write('#!/bin/bash\n')
        for item in files:
            handle.write("sbatch " + item + "\n")


def generate_slurm_scripts_nn_rf(clear_old_scripts=False, test=False):
    """
    Function for generating slurm scripts for uplift-rf grid search experiment.

    1. By default, don't empty tmp.
    2. Create a bunch of slurm_{}.job's into ./tmp/
    3. Done. The actual jobs are submitted by sbatch ./tmp/*.job

    Args:
    clear_old_scripts (bool): If true, deletes old scripts from ./tmp).
    test (bool): Generate slurm-scripts also for test dataset.

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

    if test:
        datasets = ['test']
    else:
        # Just skip Hillstrom. No point in using that for undersampling.
        #datasets = ['hillstrom', 'starbucks', 'voter']  #['zenodo']  #['voter']  #['criteo1', 'criteo2']  # , 'hillstrom']
        datasets = ['zenodo']

    configs = [  # Note that uplift-rf and uplift neural net will need separate k-selection.
        # Baselines:
        ['none', 'uplift_rf', 'none', './results/baselines.csv'],  # Ok
        ['none', 'uplift_neural_net', 'none', './results/baselines.csv'],  # Not ok.
        # k-undersampling with 
        ['k_undersampling', 'uplift_rf', 'div_by_k', './results/k_undersampling_uplift_rf_div_by_k.csv'],  # >1 not ok
        ['k_undersampling', 'uplift_neural_net', 'div_by_k', './results/k_undersampling_uplift_neural_net_div_by_k.csv'],  # 1 not ok
        # k-undersampling with calibration
        ['k_undersampling', 'uplift_rf', 'isotonic', './results/k_undersampling_uplift_rf_isotonic.csv'],  # >1 not ok
        ['k_undersampling', 'uplift_neural_net', 'isotonic', './results/k_undersampling_uplift_neural_net_isotonic.csv'],  # 1 not ok
        # Naive
        ['naive_undersampling', 'uplift_rf', 'isotonic', './results/naive_undersampling_uplift_rf_isotonic.csv'],  # >1 not ok.
        ['naive_undersampling', 'uplift_neural_net', 'isotonic', './results/naive_undersampling_uplift_neural_net_isotonic.csv']  # Ok
        # dc-undersampling + calibration (a second baseline)
    ]

    for dataset in datasets:
        for item in configs:
            assert item[1] == 'uplift_rf' or item[1] == 'uplift_neural_net',\
                "Item[1] in input should be 'uplift_rf' or 'uplift_neural_net'"
            # Set values to iterate over
            if item[0] == 'none':
                assert item[2] == 'none', "If undersampling is 'none', correction must be 'none'."
                # This is a simple baseline.
                k_values = [1]
            else:
                k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256]

            for k in k_values:
                # Set "mem" depending on what values k has.
                # This scheme did not work well for RF, though.
                if item[1] == 'uplift_neural_net':
                    # Neural nets worked well with the settings below. With
                    # the time limit of 23 hours, the neural nets will work
                    # well also for k=1. 
                    if k == 1:
                        time = '23:59:59'
                        mem = '64G'
                        part = "medium"
                    elif k > 1 and k <= 4:
                        time = '8:00:00'
                        mem = '16G'
                        part = 'short'
                    elif k > 4:
                        time = '1:00:00'
                        mem = '8G'
                        part = "short"
                    elif k is None:
                        # k-selection handled internally, needs time
                        time = "8:00:00"
                        mem = "16G"
                        part = "medium"
                elif item[1] == 'uplift_rf':
                    # Set suitable values for uplift RF.
                    time = '23:59:59'
                    mem = '64G'
                    part = "medium"
                else:
                    print("{} not a valid model.".format(item[1]))
                    raise ValueError

                undersampling = item[0]
                model = item[1]
                correction = item[2]
                output_file = item[3]
                # k is k.
                if model == 'uplift_neural_net':
                    # Neural net need gpu allocation and gpu queue.
                    # Reserve 4 CPU cores for faster dataloading.
                    part = ' gpu'
                    # Code contained #SBATCH --constraint=v100
                    text = """#!/bin/bash
#SBATCH --job-name={0}_{1}_{2}_{3}_k{8}
#SBATCH -o ./slurm_out/result_{0}_{1}_{2}_{3}_k{8}.txt
#SBATCH -p{6}
#SBATCH -c 4
#SBATCH -M ukko
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --mem-per-cpu={5}
export PYTHONPATH=$PYTHONPATH:.
srun hostname
srun sleep 5
srun python -m experiments.undersampling_experiments {0} {1} {2} {3} {7} {8}
""".format(dataset, undersampling, model, correction, time, mem, part, output_file, k)
                else:
                    # Model assumed to be uplift_rf
                    text = """#!/bin/bash
#SBATCH --job-name={0}_{1}_{2}_{3}_k{8}
#SBATCH -o ./slurm_out/result_{0}_{1}_{2}_{3}_k{8}.txt
#SBATCH -p{6}
#SBATCH -c 1
#SBATCH -M ukko
#SBATCH -t {4}
#SBATCH --mem={5}
source ~/.bashrc
export PYTHONPATH=$PYTHONPATH:.
srun hostname
srun sleep 60
srun python -m experiments.undersampling_experiments {0} {1} {2} {3} {7} {8}
""".format(dataset, undersampling, model, correction, time, mem, part, output_file, k)

                # Write text to file:
                tmp_filename = './tmp/slurm_{0}_{1}_{2}_{3}_k{8}.job'.format(
                    dataset, undersampling, model, correction,
                    time, mem, part, output_file, k)
                with open(tmp_filename, 'w') as handle:
                    handle.write(text)
                # Add filename to list to later write to bash-script.
                files.append(tmp_filename)
                # Also change the output file and perhaps even dir.
                print(text)
    # Add all to a bash-script to submit slurm-jobs:    
    # # Keep track of files in list and write to bash-script:
    with open('./tmp/bash_script.sh', 'a') as handle:
        handle.write('#!/bin/bash\n')
        for item in files:
            handle.write("sbatch " + item + "\n")



if __name__ == '__main__':
    # Generate slurm-scripts
    # Allow script to clear ./tmp/
    generate_slurm_scripts_dc_cvt(True)
    # Scripts already cleared by previous.
    generate_slurm_scripts_nn_rf(False)
