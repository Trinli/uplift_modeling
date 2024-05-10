"""
This file generates a parameter file for slurm-parameters for
grid search for split undersampling.
"""

import os


def generate_slurm_parameters(model='uplift_rf',
                              parameters_file='./tmp/parameters.txt'):
    """
    Function for generating parameters for slurm script
    for uplift-rf grid search experiment.

    Args:
    model (str): {'uplift_rf', 'dc_lr'}. Only these two models currently
     support the analytic correction.
    parameters_file (str): File name of file to write parameters to.
    """
    # Check if tmp exists:
    if not os.path.exists('./tmp/'):
        # Create path:
        os.mkdir('./tmp/')
    # Create new jobs:
    # Also add combinations for special cases (k-undersampling included as
    # k_c and k_t vectors are identical). 
    # Specia case p_t == p_c
    # New scripts with p_t* == p_c*
    r = 1.548946888639783
    k_t = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
    k_c = [r * item for item in k_t] + k_t

    parameters_file = open(parameters_file, 'w')
    dataset = 'criteo2'  # Uses criteo2_1to1.csv
    for kt in k_t:
        for kc in k_c:
            # Generate parameter line
            # srun python crf_experiment.py criteo2_1to1.csv ./datasets/criteo2/ 1.0 {0} {1} {2}
            # python -m experiments.crf_experiment criteo2_1to1.csv ./datasets/criteo2/ split_undersampling uplift_rf analytic
            # 1,3 ./results/grid_search_rf.csv
            txt = dataset + ' split_undersampling ' + model
            txt += ' analytic' + ' ./results/criteo2_' + model + '_grid_search.csv ' + str(kt) + ',' + str(kc) + '\n'
            parameters_file.write(txt)
    print("Slurm job parameters for {} and {} created successfully".format(model, dataset))

