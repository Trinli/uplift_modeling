"""
Script for generating slurm scripts for extended undersampling experiments.
Scripts will go into ./tmp/

This contains everything except for split undersampling experiments.
I.e. baselines, k-undersampling, and naive (+dc-undersampling) with
all models and none, div-by-k, and isotonic regression for calibration.
"""
import os
from pathlib import Path
from itertools import product


def generate_slurm_scripts(clear_old_scripts=False):
    """
    Function for generating slurm scripts for uncertainty experiments
    with Dirichlet-based Gaussian Process.

    1. Check whether ./tmp/ exists. Create if not.
    2. Remove *.job from ./tmp/
    3. Create a bunch of slurm_{}.job's into ./tmp/
    4. Done. The actual jobs are submitted by sbatch ./tmp/*.job

    Misc:
    -Perhaps check how long it takes to do a job with k_t or k_c = 1
     and run them separately? Will need more time at least.
    """
    # Check if tmp exists:
    if not os.path.exists('./tmp/'):
        # Create path:
        os.mkdir('./tmp/')
    # Remove *.job's from tmp
    # os.remove('./tmp/*.job')
    if clear_old_scripts:
        for p in Path("./tmp/").glob("*.job"):
            p.unlink()

    # The GP should _not_ be able to handle a training set of 1000 * 2^13 = 8M
    # Criteo - GP cannot run (OOM with like 20k observations, almost no positive)
    conf = [{'data': 'criteo2', 'model': 'gp', 'sizes': [],
             'honest': False, 'undersampling': False},
            {'data': 'criteo2', 'model': 'tree', 'sizes': [100000 * 2 ** i for i in range(8)],
             'max_leafs': 81, 'honest': True, 'undersampling': True},
            {'data': 'criteo2', 'model': 'tree', 'sizes': [100000 * 2 ** i for i in range(8)],
             'max_leafs': 81, 'honest': True, 'undersampling': False},
            {'data': 'hillstrom', 'model': 'gp', 'sizes': [500 * 2 ** i for i in range(6)] + [31959],
             'honest': False, 'undersampling': False},
            {'data': 'hillstrom', 'model': 'tree', 'sizes': [500 * 2 ** i for i in range(6)] + [31959],
             'max_leafs': 34, 'honest': True, 'undersampling': True},
            {'data': 'hillstrom', 'model': 'tree', 'sizes': [500 * 2 ** i for i in range(6)] + [31959],
             'max_leafs': 34, 'honest': True, 'undersampling': False},
            {'data': 'starbucks', 'model': 'gp', 'sizes': [500 * 2 ** i for i in range(7)],
             'honest': False, 'undersampling': False},
            {'data': 'starbucks', 'model': 'tree', 'sizes': [500 * 2 ** i for i in range(9)],
             'max_leafs': 12, 'honest': True, 'undersampling': True},
            {'data': 'starbucks', 'model': 'tree', 'sizes': [500 * 2 ** i for i in range(9)],
             'max_leafs': 12, 'honest': True, 'undersampling': False}
    ]

    a_eps_list = [0.5, 0.1, 0.01, 0.001, 0.0001]
    # Store filenames to write to bash script later:
    files = []

    for item in conf:
        dataset = item['data']
        model = item['model']
        sizes = item['sizes']
        honest = item['honest']
        undersampling = item['undersampling']
        try:
            max_leaf = item['max_leafs']
        except:
            max_leaf = None
        """
        Setups

        Hillstrom:
        GP - all sizes up til max data
         -"Cuda unavailable". Run experiments to size 8k.
         -Almost done.
        Tree - non-honest version, #leafs: ?  # PICK A NUMBER!!
         -Should run just fine.

        Starbucks:
        GP - all sizes up to 20k
        Tree - honest (no undersampling)
         -Not enough positive observations in some leafs? Parameters for beta-distribution
          invalid?

        Criteo2:
        -conversion label (should we try visit?)
        -#leafs: ?
        GP: none
        Tree: honest & honest with undersampling

        Max leaf size:
        -How do we deal with train/val?
        -Right now: smaller of treatment/control, #positive / max_leafs = 50
        --i.e. on average, leafs will contain 50 observations
        """
        if model == 'tree':
            iterable = product(sizes, [0])
        elif model == 'gp':
            iterable = product(sizes, a_eps_list)
        for size, a_eps in iterable:
            # Model is gp or tree
            time = "23:59:59"
            mem = "64G"
            if model == 'gp':
                part = "gpu"
            elif model == 'tree':
                part = 'medium'
            else:
                raise Exepction('No valid model selected.')
            if model == 'tree':
                text = """#!/bin/bash
#SBATCH --job-name={1}_{0}_{5}
#SBATCH -o ./slurm_out/result_{1}_{0}_{5}.txt
#SBATCH -M ukko
#SBATCH -c 1
#SBATCH -t {2}
#SBATCH --mem={3}
#SBATCH -p {4}
export PYTHONPATH=$PYTHONPATH:.
srun hostname
srun sleep 5
srun python -m experiments.uncertainty_experiments {1} {0} {5} {6} {7} {8}
""".format(dataset, model, time, mem, part, size, max_leaf, honest, undersampling)
            elif model == 'gp':
                text = """#!/bin/bash
#SBATCH --job-name={1}_{0}_{5}
#SBATCH -o ./slurm_out/result_{1}_{0}_{5}.txt
#SBATCH -M ukko
#SBATCH -c 1
#SBATCH -t {2}
#SBATCH --mem={3}
#SBATCH -p {4}
#SBATCH --gres=gpu:1
#SBATCH --constraint=[v100]
export PYTHONPATH=$PYTHONPATH:.
srun hostname
srun sleep 5
srun python -m experiments.uncertainty_experiments {1} {0} {5} {6}
""".format(dataset, model, time, mem, part, size, a_eps)

            # Write text to file:
            if model == 'tree':
                tmp_filename = './tmp/slurm_{1}_{0}_{2}_{3}_{4}.job'.format(
                    dataset, model, size, item['honest'], item['undersampling'])
            elif model == 'gp':
                tmp_filename = './tmp/slurm_{1}_{0}_{2}_{3}.job'.format(
                    dataset, model, size, a_eps)
            with open(tmp_filename, 'w') as handle:
                handle.write(text)
            # Add filename to list to later write to bash-script.
            files.append(tmp_filename)
            # Also change the output file and perhaps even dir.
            print(text)
    # Add all to a bash-script to submit slurm-jobs:    
    # Keep track of files in list and write to bash-script:
    with open('./tmp/bash_script.sh', 'w') as handle:
        handle.write('#!/bin/bash\n')
        for item in files:
            handle.write("sbatch " + item + "\n")


if __name__ == '__main__':
    # Generate slurm-scripts
    # Allow script to clear ./tmp/
    generate_slurm_scripts(True)
