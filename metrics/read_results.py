"""
The undersampling experiments were partly run in parallel and the results
were written into files depending on parallelization and some other parameters.
This file reads in the results and processes them appropriately.

Per dataset
    Per model-combination (maybe use same grid as in planning stage, model vs. undersampling + cal)
    17 + 2 combinations (2 with grid-search)
        Validation set results: AUUC vs k and best k
            table
            visualization (AUUC vs. k per model)
        Tesing set results: AUUC
            Table
"""
import csv
import re
import numpy as np

# 1. Read in csv
# -there might be a list of them. Maybe list them, or
# read in all in some directory.
# 2. Sort the appropriately
# some files needs the max to be found in the validation set
# results, and after that the corresponding max is looked for
# in the testing set file.

# Topics of interest
# -validation set performance for different model combinations over k
# -testing set performance for best model (on validation set) and model combinations
# -identify is some runs crashed

# Result files renamed and moved to folder at this point:
path = './run_2_extended_experiments/'
#path = './results/voter/'
files = [
        # k-undersampling + some model + div_by_k
         ['k_undersampling_dc_lr_div_by_k.csv',
          'k_undersampling_dc_lr_div_by_k.csv_testing_set.csv'],
         ['k_undersampling_cvt_lr_div_by_k.csv',
          'k_undersampling_cvt_lr_div_by_k.csv_testing_set.csv'],
         ['k_undersampling_uplift_neural_net_div_by_k.csv',
          'k_undersampling_uplift_neural_net_div_by_k.csv_testing_set.csv'],
         ['k_undersampling_uplift_rf_div_by_k.csv', 
          'k_undersampling_uplift_rf_div_by_k.csv_testing_set.csv'],
        # k-undersampling + some model + isotonic
         ['k_undersampling_dc_lr_isotonic.csv',
          'k_undersampling_dc_lr_isotonic.csv_testing_set.csv'],
         ['k_undersampling_cvt_lr_isotonic.csv',
          'k_undersampling_cvt_lr_isotonic.csv_testing_set.csv'],
         ['k_undersampling_uplift_neural_net_isotonic.csv',
          'k_undersampling_uplift_neural_net_isotonic.csv_testing_set.csv'],
         ['k_undersampling_uplift_rf_isotonic.csv',
          'k_undersampling_uplift_rf_isotonic.csv_testing_set.csv'],
         # naive + some model + isotonic
         ['naive_undersampling_dc_lr_isotonic.csv',
          'naive_undersampling_dc_lr_isotonic.csv_testing_set.csv'],
         ['naive_undersampling_cvt_lr_isotonic.csv',
          'naive_undersampling_cvt_lr_isotonic.csv_testing_set.csv'],
         ['naive_undersampling_uplift_neural_net_isotonic.csv',
          'naive_undersampling_uplift_neural_net_isotonic.csv_testing_set.csv'],
         ['naive_undersampling_uplift_rf_isotonic.csv',
          'naive_undersampling_uplift_rf_isotonic.csv_testing_set.csv'],
         # Split undersampling
         ['results_dc_grid_search_criteo1.csv',  
          'results_dc_grid_search_criteo1.csv_testing_set.csv'],  # These currently contain no criteo2-results.
         ['results_rf_grid_search_criteo1.csv',  # These currently contain no criteo2-results.
          'results_rf_grid_search_criteo1.csv_testing_set.csv'],  # These currently contain no criteo2-results.
         #['../results_uplift_dc/grid_search/results.csv', ...],  # This is just training set results.
         #['../results_uplift_rf/grid_search/results.csv', ...],  # No testing set results?
         # Baseline (dc-underesampling + lr)
         ['dc_undersampling_naive_dc_none.csv',
          'dc_undersampling_naive_dc_none.csv_testing_set.csv'],
         # baselines (none + some model + none)
         ['baselines.csv_testing_set.csv',
          'baselines.csv_testing_set.csv'],  # Just baselines
         ]


def read_file(tmp_file, path=path):
    """
    Function for reading in result files.
    """
    lines = []
    with open(path + tmp_file) as handle:
        file_reader = csv.reader(handle, delimiter=';')
        for line in file_reader:
            lines.append(line)
    return lines


def parse_result_files(dataset='criteo1', path=path):
    """
    Massive function for parsing results from files named as in the files-list.
    The files-list contains items that are lists of two files. The first one
    should contain the validation set results as written by metrics.uplift_metrics.UpiftMetrics,
    the second one testing set results.
    This function will traverse all items except for the last two, searching for the best
    AUUC in the validation set and then storing the corresponding testing set AUUC in
    a list. The last two items are baselines and contain nothing but testing set results for
    single models (4 in baselines). These are added to the list as well.

    Args:
    dataset (str): {'criteo1', 'criteo2'}. Pick which dataset you want results for.
    """
    # Populate some matrix with the results, one per dataset.
    criteo_results = [
        # Dataset, undersampling, model, correction, auuc, best parameters (k or k's)
        # Note that ordering does _not_ match right now.
        ['criteo', 'none', 'dc_lr', 'none'],
        ['criteo', 'none', 'cvt_lr', 'none'],
        ['criteo', 'none', 'uplift_neural_net', 'none'],
        ['criteo', 'none', 'uplift_rf', 'none'],
        ['criteo', 'k_undersampling', 'dc_lr', 'div_by_k'],
        ['criteo', 'k_undersampling', 'cvt_lr', 'div_by_k'],
        ['criteo', 'k_undersampling', 'uplift_nn', 'div_by_k'],
        ['criteo', 'k_undersampling', 'uplift_rf', 'div_by_k'],
        ['criteo', 'k_undersampling', 'dc_lr', 'isotonic'],
        ['criteo', 'k_undersampling', 'cvt_lr', 'isotonic'],
        ['criteo', 'k_undersampling', 'uplift_nn', 'isotonic'],
        ['criteo', 'k_undersampling', 'uplift_rf', 'isotonic'],
        ['criteo', 'naive', 'dc_lr', 'isotonic'],
        ['criteo', 'naive', 'cvt_lr', 'isotonic'],
        ['criteo', 'naive', 'uplift_nn', 'isotonic'],
        ['criteo', 'naive', 'uplift_rf', 'isotonic'],
        ['criteo', 'grid_search', 'dc_lr', 'analytic'],
        ['criteo', 'grid_search', 'uplift_rf', 'analytic'],
        ['criteo', 'dc_undersampling', 'naive_dc', 'isotonic'],
    ]

    # First baselines:
    tmp = read_file(files[-1][0], path=path)
    print("=" * 40)
    print("Baselines:")
    for line in tmp:
        print("{}, {}, {}, {}, {}".format(line[3], line[1], line[4], line[6], line[7]))
        if dataset in line[1].lower():  # Correct dataset
            for i, item in enumerate(criteo_results[:4]):  # First four are baseline models
                if line[3] == item[2]:  # If models match
                    criteo_results[i].append(line[4])  # Add details (k)
                    criteo_results[i].append(line[7])  # auuc
                    criteo_results[i].append(line[8])  # improvement to random
                    criteo_results[i].append(line[10])  # EUCE

    # Next dc-undersampling + lr (second baseline)
    tmp = read_file(files[-2][1], path=path)  # Testing set in [1]
    print("=" * 40)
    print("DC-undersampling")
    for line in tmp:
        print("{}, {}, {}, {}, {}".format(line[0], line[1], line[4], line[6], line[7]))
        print(line)
        if dataset in line[1].lower():  # Correct dataset
            item = criteo_results[-1]  # Last one is dc-undersampling + lr (baseline)
            if line[3] == item[2]:  # If models match
                criteo_results[-1].append(line[4])  # Add details (k)
                criteo_results[-1].append(line[7])  # auuc
                criteo_results[-1].append(line[8])  # improvement to random
                criteo_results[-1].append(line[10])  # EUCE

    # Read in file [i][0] to find max. Find corresponding line in [i][1]. Maybe
    # remove header rows that seem to be all over the place.
    # Print model name, best k, and testing set metrics.
    print("=" * 40)
    print("Other models (note criteo2 hack)")
    for j, item in enumerate(files[:-2]):
        validation_set_file = item[0]
        lines = read_file(validation_set_file, path=path)
        # Find max AUUC and corresponding k in validation set
        # AUUC at [7], improvement to rand at [8]
        # dataset at [1], k at [4]
        tmp_auuc = -1.0
        # Reading in training set to find best parameters
        for line in lines:
            if dataset in line[1].lower():
                # I.e. we are working on *this dataset:
                if tmp_auuc <= float(line[7]):  # Taking the "last" good model as best if there is a tie.
                    # Keep best k-values
                    best_val_line = line
                    best_k = line[4]
                    tmp_auuc = float(line[7])
        # Print the best validation set results
        # Test details, dataset, k, improvement over random, auuc
        print("=" * 40)
        print("Validation set results:")
        print("{r[3]}, {r[1]}, {r[4]}, {r[6]}, {r[7]}".format(r=best_val_line))
        print("Lines in validation set: {}".format(len(lines)))
        testing_set_file = item[1]
        lines_test = read_file(testing_set_file, path=path)
        # Find line that matches dataset and best_k
        for line in lines_test:
            # criteo_results has the same indexing as files. We should be picking the one line that matches the k defined in best_k above.
            if line[4] == best_k:  # If best_k from validation set equals the best k in the row of the testing set file, then store:
                if dataset in line[1].lower():
                    # +4 to leave the first four baselines untouched
                    best_test_line = line
                    if len(criteo_results[j + 4]) == 4:
                        print("Adding results to array...")
                        criteo_results[j + 4].append(line[4])  # Add details (k)
                        criteo_results[j + 4].append(line[7])  # auuc
                        criteo_results[j + 4].append(line[8])  # improvement to random
                        criteo_results[j + 4].append(line[10])  # EUCE
                    else:
                        # Replace results with last ones:
                        print("Result-file contains duplicates. Using last matching row.")
                        criteo_results[j + 4][4] = line[4]  # Replace k
                        criteo_results[j + 4][5] = line[7]  # Replace AUUC
                        criteo_results[j + 4][6] = line[8]  # Replace improvement to random
                        criteo_results[j + 4][7] = line[10]  # EUCE
        print("Testing set results:")
        print("{r[3]}, {r[1]}, {r[4]}, {r[6]}, {r[7]}".format(r=best_test_line))
        print("Lines in testing set: {}".format(len(lines_test)))
    return criteo_results


def zenodo(dataset='zenodo'):
    """
    Hacky function for reading zenodo results

    Initialise criteo_results list above...
    Set path as desired (needs to be set globally)
    """
    path = '/Users/ottonyberg/Desktop/uplift/results/zenodo/zenodo_02/rf_grid/'
    # path = './DAMI3/results_uplift_dc/'  # For new results for new datasets
    # path = './DAMI3/results_uplift_rf/'  # For new results for new datasets
    files = [['results.csv', 'results.csv_testing_set.csv']]
    #criteo_results = []
    for j, item in enumerate(files):
        validation_set_file = item[0]
        lines = read_file(validation_set_file)
        # Find max AUUC and corresponding k in validation set
        # AUUC at [7], improvement to rand at [8]
        # dataset at [1], k at [4]
        tmp_auuc = -1.0
        # Reading in training set to find best parameters
        for line in lines:
            if dataset in line[1].lower():
                # I.e. we are working on *this dataset:
                if tmp_auuc <= float(line[7]):  # Taking the "last" good model as best if there is a tie.
                    # Keep best k-values
                    best_val_line = line
                    best_k = line[4]
                    tmp_auuc = float(line[7])
        # Print the best validation set results
        # Test details, dataset, k, improvement over random, auuc
        print("=" * 40)
        print("Validation set results:")
        print("{r[3]}, {r[1]}, {r[4]}, {r[6]}, {r[7]}".format(r=best_val_line))
        print("Lines in validation set: {}".format(len(lines)))
        testing_set_file = item[1]
        print(1)
        lines_test = read_file(testing_set_file)
        print(2)
        # Find line that matches dataset and best_k
        for line in lines_test:
            print(3)
            # criteo_results has the same indexing as files. We should be picking the one line that matches the k defined in best_k above.
            if line[4] == best_k:  # If best_k from validation set equals the best k in the row of the testing set file, then store:
                print(4)
                if dataset in line[1].lower():
                    print(5)
                    # +4 to leave the first four baselines untouched
                    best_test_line = line
                    if len(criteo_results[j]) == 4:
                        print(6)
                        criteo_results[j].append(line[4])  # Add details (k)
                        criteo_results[j].append(line[7])  # auuc
                        criteo_results[j].append(line[8])  # improvement to random
                        criteo_results[j].append(line[10])  # EUCE
                    else:
                        print(7)
                        # Replace results with last ones:
                        print("Result-file contains duplicates. Using last matching row.")
                        criteo_results[j][4] = line[4]  # Replace k
                        criteo_results[j][5] = line[7]  # Replace AUUC
                        criteo_results[j][6] = line[8]  # Replace improvement to random
                        criteo_results[j][7] = line[10]  # EUCE
        print("Testing set results:")
        print("{r[3]}, {r[1]}, {r[4]}, {r[6]}, {r[7]}".format(r=best_test_line))
        print("Lines in testing set: {}".format(len(lines_test)))



def criteo2_hack(criteo_results):
    """
    This function is a quick hack for the criteo2-results where I did not
    estimate testing set metrics for _all_ models (with all parameters).
    The files below should have only the best k testing set output in them.
    """
    # If dataset is criteo2, then overwrite (is there anything) the results stored earlier.
    # DC-grid search results:
    lines = read_file('results_dc_grid_search.csv_testing_set.csv')
    # Results in line [1]
    line = lines[1]
    criteo_results[16].append(line[4])
    criteo_results[16].append(line[7])
    criteo_results[16].append(line[8])
    criteo_results[16].append(line[10])
    lines = read_file('results_rf_grid_search.csv_testing_set.csv')
    # Results in line [1]
    line = lines[1]
    criteo_results[17].append(line[4])
    criteo_results[17].append(line[7])
    criteo_results[17].append(line[8])
    criteo_results[17].append(line[10])
    return criteo_results


def find_best_k():
    """
    Function for finding the best k_c and k_t in defined files. 
    """
    # Find max k_t and k_c for grid search (both with Criteo1 and 2, and DC-LR and Uplift RF)
    # Delete this? Nope. This is used for criteo2 where the experiments did not calculate
    # testing set metrics for all models.
    lines = read_file('../results_uplift_dc/grid_search/results.csv')
    best_auuc = 0
    for line in lines:
        try:
            if best_auuc < float(line[7]):
                best_auuc = float(line[7])
                best_line = line
        except ValueError:
            # Skip header row (that might be embedded somewhere)
            pass
    print("Best validation set results for grid-search DC")
    print(best_line)

    # This does not store anything to the list.
    # Delete? Nope. Finds best parameters for criteo2 experiments where
    # the testing set metrics were not estimated for all models.
    lines = read_file('../results_uplift_rf/grid_search/results.csv')
    best_auuc = 0
    for line in lines:
        try:
            if best_auuc < float(line[7]):
                best_auuc = float(line[7])
                best_line = line
        except ValueError:
            # Skip header row (that might be embedded somewhere)
            pass
    print("Best validation set results for grid-search Uplift RF")
    print(best_line)


def print_results(results, ignore_isotonic=False):
    # Print the result table:
    print("=" * 40)
    print("Testing set results:")
    if ignore_isotonic:
        for row in results[:8]:
            print(row)
        for row in results[12:]:
            print(row)
    else:
        for row in results:
            print(row)


def print_latex(results, metric='auuc'):
    """
    Print in a format that can be copy-pasted into a latex file.

    Args:
    results (list): List of results to be formatted
    metric (str): 'auuc' or 'k'. With 'auuc', AUUC is printed
     in the latex code. With 'k', the optimal k-vaues are
     printed.

    Notes:
    None of the values are stored as floats in criteo_results,
    it is all strings. Consequently formatting does not work
    correctly for scientific formats (e.g '5.633213e-7')
   """
    def floatify(item):
        try:
            tmp = float(item)
            return tmp  # Needs to break here.
        except ValueError:
            # Argh. It is not stored as a list, it is stored as a string,
            # e.g. "'[4.0, 12.39..]'". Regex
            #re... "\[ [\d*]\.[\d*],"
            # ", ... \]"
            pass
        try:
            #p = re.compile('[\d]*\.[\d]*')  # Matches e.g. "41.23"
            p = re.compile('[0-9]+.[0-9]+')
            tmp = p.findall(item)
            if len(tmp) == 0:
                # Previous match did not produce expected results. Try differently:
                q = re.compile('[0-9]+')  # Matches ints.
                tmp = q.findall(item)
            tmp = [float(tmp[0]), float(tmp[1])]
        except IndexError:
            tmp = None
        return tmp
    results = [[floatify(item) for item in row] for row in results]
    # Multiply all AUUC vales by 1000 to get readable format
    def multiply(item, i, factor=1000):
        if i == 5:
            return factor * item
        else:
            return item
    results = [[multiply(item, i) for i, item in enumerate(row)] for row in results]
    header = """\\begin{table}[h]
%\\begin{center}
\\begin{minipage}{174pt}
"""
    if metric == 'auuc':
        header += """\caption{mAUUC on dataset ...}\label{some_table}
"""
    elif metric == 'k':
        header += """\caption{Optimal k-values on dataset ...}\label{some_table}
"""
    elif metric == 'euce':
        header += """\caption{EUCE on dataset ...}\label{some_table}
"""
    header +="""\\begin{tabular}{@{}lllll@{}}
\\toprule
 & DC-LR & CVT-LR & Uplift RF & Uplift NN \\\\
\midrule"""

    if metric == 'auuc':
        # The middle- section should ge generated from the results
        # Note that the RF and NN indices are in reversed order for printing
        # (list has NN first, result print RF first)
        middle_section = \
"""No undersampling & {r0[5]:.3f} & {r1[5]:.3f} & {r3[5]:.3f} & {r2[5]:.3f}\\\\
Classic undersampling & {r18[5]:.3f} & n/a & n/a & n/a\\\\
Naive undersampling & {r12[5]:.3f} & {r13[5]:.3f} & {r15[5]:.3f} & {r14[5]:.3f}\\\\
Stratified undersampling & {r4[5]:.3f} & {r5[5]:.3f} & {r7[5]:.3f} & {r6[5]:.3f}\\\\
Split undersampling & {r16[5]:.3f} & n/a & {r17[5]:.3f} & n/a\\\\
""".format(r0=results[0], r1=results[1], r2=results[2], r3=results[3],
            r4=results[4], r5=results[5], r6=results[6], r7=results[7],
            # 8-11 are k-undersamping + isotonic. Skip.
            #r8=results[8], r9=results[9], r10=results[10], r11=results[11],
            r12=results[12], r13=results[13], r14=results[14], r15=results[15],
            #r16=[0,0,0,0,0,0.0], r17=[0,0,0,0,0,0.0],
            r16=results[16], r17=results[17],   # Waiting for the experiments to finish.
            r18=results[18])

    elif metric == 'k':  # Classic-undersampling should take _two_ values!! It takes one now.
        middle_section = \
"""No undersampling & {r0[4]:.0f} & {r1[4]:.0f} & {r3[4]:.0f} & {r2[4]:.0f}\\\\
Classic undersampling & {r18[4]} & n/a & n/a & n/a\\\\
Naive undersampling & {r12[4]:.0f} & {r13[4]:.0f} & {r15[4]:.0f} & {r14[4]:.0f} \\\\
Stratified undersampling & {r4[4]:.0f} & {r5[4]:.0f} & {r7[4]:.0f} & {r6[4]:.0f}\\\\
Split undersampling & {r16[4]} & n/a & {r17[4]} & n/a\\\\
""".format(r0=results[0], r1=results[1], r2=results[2], r3=results[3],
            r4=results[4], r5=results[5], r6=results[6], r7=results[7],
            # 8-11 are k-undersamping + isotonic. Skip.
            #r8=results[8], r9=results[9], r10=results[10], r11=results[11],
            r12=results[12], r13=results[13], r14=results[14], r15=results[15],
            #r16=[0,0,0,0,0,0.0], r17=[0,0,0,0,0,0.0],
            r16=results[16], r17=results[17],   # Waiting for the experiments to finish.
            r18=results[18])

    elif metric == 'euce':
        # The middle- section should ge generated from the results
        # Note that the RF and NN indices are in reversed order for printing
        # (list has NN first, result print RF first)
        middle_section = \
"""No undersampling & {r0[7]:.5f} & {r1[7]:.5f} & {r3[7]:.5f} & {r2[7]:.5f}\\\\
Classic undersampling & {r18[7]:.5f} & n/a & n/a & n/a\\\\
Naive undersampling & {r12[7]:.5f} & {r13[7]:.5f} & {r15[7]:.5f} & {r14[7]:.5f}\\\\
K-undersampling & {r4[7]:.5f} & {r5[7]:.5f} & {r7[7]:.5f} & {r6[7]:.5f}\\\\
Split-undersampling & {r16[7]:.5f} & n/a & {r17[7]:.5f} & n/a\\\\
""".format(r0=results[0], r1=results[1], r2=results[2], r3=results[3],
            r4=results[4], r5=results[5], r6=results[6], r7=results[7],
            # 8-11 are k-undersamping + isotonic. Skip.
            #r8=results[8], r9=results[9], r10=results[10], r11=results[11],
            r12=results[12], r13=results[13], r14=results[14], r15=results[15],
            #r16=[0,0,0,0,0,0.0], r17=[0,0,0,0,0,0.0],
            r16=results[16], r17=results[17],   # Waiting for the experiments to finish.
            r18=results[18])

    footer = \
"""\\botrule
\end{tabular}
\end{minipage}
%\end{center}
\end{table}"""
    print(header)
    print(middle_section)
    print(footer)


def print_latex_w_std(results, std_results, metric='auuc'):
    """
    Print in a format that can be copy-pasted into a latex file.

    Args:
    results (list): List of results to be formatted
    std_results (list): List of standard deviations
    metric (str): 'auuc' or 'k'. With 'auuc', AUUC is printed
     in the latex code. With 'k', the optimal k-vaues are
     printed.

    Notes:
    None of the values are stored as floats in criteo_results,
    it is all strings. Consequently formatting does not work
    correctly for scientific formats (e.g '5.633213e-7')
    """
    def floatify(item):
        try:
            tmp = float(item)
            return tmp  # Needs to break here.
        except ValueError:
            # Argh. It is not stored as a list, it is stored as a string,
            # e.g. "'[4.0, 12.39..]'". Regex
            #re... "\[ [\d*]\.[\d*],"
            # ", ... \]"
            pass
        try:
            #p = re.compile('[\d]*\.[\d]*')  # Matches e.g. "41.23"
            p = re.compile('[0-9]+.[0-9]+')
            tmp = p.findall(item)
            if len(tmp) == 0:
                # Previous match did not produce expected results. Try differently:
                q = re.compile('[0-9]+')  # Matches ints.
                tmp = q.findall(item)
            tmp = [float(tmp[0]), float(tmp[1])]
        except IndexError:
            tmp = None
        return tmp
    # I think everything is floats alreadey (?)
    #results = [[floatify(item) for item in row] for row in results]
    # Multiply all AUUC vales by 1000 to get readable format
    def multiply(item, i, factor=1000):
        if i == 5:
            return factor * item
        else:
            return item
    results = [[multiply(item, i) for i, item in enumerate(row)] for row in results]
    std_results = [[multiply(item, i) for i, item in enumerate(row)] for row in std_results]
    header = """\\begin{table}[h]
%\\begin{center}
\\begin{minipage}{174pt}
"""
    if metric == 'auuc':
        header += """\caption{mAUUC on dataset ..., standard deviation in parenthesis.}\label{some_table}
"""
    elif metric == 'k':
        header += """\caption{Optimal k-values on dataset ..., standard deviation in parenthesis.}\label{some_table}
"""
    elif metric == 'euce':
        header += """\caption{EUCE on dataset ..., standard deviation in parenthesis.}\label{some_table}
"""
    header +="""\\begin{tabular}{@{}lllll@{}}
\\toprule
 & DC-LR & CVT-LR & Uplift RF & Uplift NN \\\\
\midrule"""
    if metric == 'auuc':
        # The middle- section should ge generated from the results
        # Note that the RF and NN indices are in reversed order for printing
        # (list has NN first, result print RF first)
        middle_section = \
"""No undersampling & {r0[5]:.3f} & {r1[5]:.3f} & {r3[5]:.3f} & {r2[5]:.3f}\\\\
 & ({s0[5]:.3f}) & ({s1[5]:.3f}) & ({s3[5]:.3f}) & ({s2[5]:.3f})\\\\
Classic undersampling & {r18[5]:.3f} & n/a & n/a & n/a\\\\
 & ({s18[5]:.3f}) & n/a & n/a & n/a\\\\
Naive undersampling & {r12[5]:.3f} & {r13[5]:.3f} & {r15[5]:.3f} & {r14[5]:.3f}\\\\
 & ({s12[5]:.3f}) & ({s13[5]:.3f}) & ({s15[5]:.3f}) & ({s14[5]:.3f})\\\\
Stratified undersampling & {r4[5]:.3f} & {r5[5]:.3f} & {r7[5]:.3f} & {r6[5]:.3f}\\\\
 & ({s4[5]:.3f}) & ({s5[5]:.3f}) & ({s7[5]:.3f}) & ({s6[5]:.3f})\\\\
Split undersampling & {r16[5]:.3f} & n/a & {r17[5]:.3f} & n/a\\\\
 & ({s16[5]:.3f}) & n/a & ({s17[5]:.3f}) & n/a\\\\
""".format(r0=results[0], r1=results[1], r2=results[2], r3=results[3],
            r4=results[4], r5=results[5], r6=results[6], r7=results[7],
            # 8-11 are k-undersamping + isotonic. Skip.
            #r8=results[8], r9=results[9], r10=results[10], r11=results[11],
            r12=results[12], r13=results[13], r14=results[14], r15=results[15],
            #r16=[0,0,0,0,0,0.0], r17=[0,0,0,0,0,0.0],
            r16=results[16], r17=results[17],   # Waiting for the experiments to finish.
            r18=results[18],
            s0=std_results[0], s1=std_results[1], s2=std_results[2], s3=std_results[3],
            s4=std_results[4], s5=std_results[5], s6=std_results[6], s7=std_results[7],
            s12=std_results[12], s13=std_results[13], s14=std_results[14], s15=std_results[15],
            s16=std_results[16], s17=std_results[17], s18=std_results[18])
    elif metric == 'k':  # Classic-undersampling should take _two_ values!! It takes one now.
        middle_section = \
"""No undersampling & {r0[4]:.0f} & {r1[4]:.0f} & {r3[4]:.0f} & {r2[4]:.0f}\\\\
 & ({s0[4]:.0f}) & ({s1[4]:.0f}) & ({s3[4]:.0f}) & ({s2[4]:.0f})\\\\
Classic undersampling & {r18[4]} & n/a & n/a & n/a\\\\
 & ({s18[4]}) & n/a & n/a & n/a\\\\
Naive undersampling & {r12[4]:.0f} & {r13[4]:.0f} & {r15[4]:.0f} & {r14[4]:.0f} \\\\
 & ({s12[4]:.3f}) & ({s13[4]:.3f}) & ({s15[4]:.3f}) & ({s14[4]:.3f}) \\\\
Stratified undersampling & {r4[4]:.0f} & {r5[4]:.0f} & {r7[4]:.0f} & {r6[4]:.0f}\\\\
 & ({s4[4]:.3f}) & ({s5[4]:.3f}) & ({s7[4]:.3f}) & ({s6[4]:.3f})\\\\
Split undersampling & {r16[4]} & n/a & {r17[4]} & n/a\\\\
 & ({s16[4]:}) & n/a & ({s17[4]:}) & n/a\\\\
""".format(r0=results[0], r1=results[1], r2=results[2], r3=results[3],
            r4=results[4], r5=results[5], r6=results[6], r7=results[7],
            # 8-11 are k-undersamping + isotonic. Skip.
            #r8=results[8], r9=results[9], r10=results[10], r11=results[11],
            r12=results[12], r13=results[13], r14=results[14], r15=results[15],
            #r16=[0,0,0,0,0,0.0], r17=[0,0,0,0,0,0.0],
            r16=results[16], r17=results[17],   # Waiting for the experiments to finish.
            r18=results[18],
            s0=std_results[0], s1=std_results[1], s2=std_results[2], s3=std_results[3],
            s4=std_results[4], s5=std_results[5], s6=std_results[6], s7=std_results[7],
            s12=std_results[12], s13=std_results[13], s14=std_results[14], s15=std_results[15],
            s16=std_results[16], s17=std_results[17], s18=std_results[18])
    elif metric == 'euce':
        # The middle- section should ge generated from the results
        # Note that the RF and NN indices are in reversed order for printing
        # (list has NN first, result print RF first)
        middle_section = \
"""No undersampling & {r0[7]:.5f} & {r1[7]:.5f} & {r3[7]:.5f} & {r2[7]:.5f}\\\\
 & ({s0[7]:.5f}) & ({s1[7]:.5f}) & ({s3[7]:.5f}) & ({s2[7]:.5f})\\\\
Classic undersampling & {r18[7]:.5f} & n/a & n/a & n/a\\\\
 & ({s18[7]:.5f}) & n/a & n/a & n/a\\\\
Naive undersampling & {r12[7]:.5f} & {r13[7]:.5f} & {r15[7]:.5f} & {r14[7]:.5f}\\\\
 & ({s12[7]:.5f}) & ({s13[7]:.5f}) & ({s15[7]:.5f}) & ({s14[7]:.5f})\\\\
K-undersampling & {r4[7]:.5f} & {r5[7]:.5f} & {r7[7]:.5f} & {r6[7]:.5f}\\\\
 & ({s4[7]:.5f}) & ({s5[7]:.5f}) & ({s7[7]:.5f}) & ({s6[7]:.5f})\\\\
Split-undersampling & {r16[7]:.5f} & n/a & {r17[7]:.5f} & n/a\\\\
 & ({s16[7]:.5f}) & n/a & ({s17[7]:.5f}) & n/a\\\\
""".format(r0=results[0], r1=results[1], r2=results[2], r3=results[3],
            r4=results[4], r5=results[5], r6=results[6], r7=results[7],
            # 8-11 are k-undersamping + isotonic. Skip.
            #r8=results[8], r9=results[9], r10=results[10], r11=results[11],
            r12=results[12], r13=results[13], r14=results[14], r15=results[15],
            #r16=[0,0,0,0,0,0.0], r17=[0,0,0,0,0,0.0],
            r16=results[16], r17=results[17],   # Waiting for the experiments to finish.
            r18=results[18],
            s0=std_results[0], s1=std_results[1], s2=std_results[2], s3=std_results[3],
            s4=std_results[4], s5=std_results[5], s6=std_results[6], s7=std_results[7],
            s12=std_results[12], s13=std_results[13], s14=std_results[14], s15=std_results[15],
            s16=std_results[16], s17=std_results[17], s18=std_results[18])

    footer = \
"""\\botrule
\end{tabular}
\end{minipage}
%\end{center}
\end{table}"""
    print(header)
    print(middle_section)
    print(footer)


def print_latex_median_and_range(results, range_results, metric='k'):
    """
    Print in a format that can be copy-pasted into a latex file.

    Args:
    results (list): List of results to be formatted
    range_results (list): List of standard deviations
    metric (str): 'auuc' or 'k'. With 'auuc', AUUC is printed
     in the latex code. With 'k', the optimal k-vaues are
     printed.

    Notes:
    None of the values are stored as floats in criteo_results,
    it is all strings. Consequently formatting does not work
    correctly for scientific formats (e.g '5.633213e-7')
    """
    # Multiply all AUUC vales by 1000 to get readable format
    def multiply(item, i, factor=1000):
        if i == 5:
            return factor * item
        else:
            return item
    results = [[multiply(item, i) for i, item in enumerate(row)] for row in results]
    range_results = [[multiply(item, i) for i, item in enumerate(row)] for row in range_results]
    header = """\\begin{table}[h]
%\\begin{center}
\\begin{minipage}{174pt}
"""
    if metric == 'k':
        header += """\caption{Medians of optimal k-values on \\texttt{dataset} of 10 runs, minimum and maximum in parenthesis.}\label{some_table}
"""
    header +="""\\begin{tabular}{@{}lllll@{}}
\\toprule
 & DC-LR & CVT-LR & Uplift RF & Uplift NN \\\\
\midrule"""
    if metric == 'k':  # Classic-undersampling should take _two_ values!! It takes one now.
        middle_section = \
"""No undersampling & {r0[4]} & {r1[4]} & {r3[4]} & {r2[4]}\\\\
 & ({s0[4]}) & ({s1[4]}) & ({s3[4]}) & ({s2[4]})\\\\
Classic undersampling & {r18[4]} & n/a & n/a & n/a\\\\
 & ({s18[4]}) & n/a & n/a & n/a\\\\
Naive undersampling & {r12[4]} & {r13[4]} & {r15[4]} & {r14[4]} \\\\
 & ({s12[4]}) & ({s13[4]}) & ({s15[4]}) & ({s14[4]}) \\\\
Stratified undersampling & {r4[4]} & {r5[4]} & {r7[4]} & {r6[4]}\\\\
 & ({s4[4]}) & ({s5[4]}) & ({s7[4]}) & ({s6[4]})\\\\
Split undersampling & {r16[4]} & n/a & {r17[4]} & n/a\\\\
 & ({s16[4]}) & n/a & ({s17[4]}) & n/a\\\\
""".format(r0=results[0], r1=results[1], r2=results[2], r3=results[3],
            r4=results[4], r5=results[5], r6=results[6], r7=results[7],
            # 8-11 are k-undersamping + isotonic. Skip.
            #r8=results[8], r9=results[9], r10=results[10], r11=results[11],
            r12=results[12], r13=results[13], r14=results[14], r15=results[15],
            #r16=[0,0,0,0,0,0.0], r17=[0,0,0,0,0,0.0],
            r16=results[16], r17=results[17],   # Waiting for the experiments to finish.
            r18=results[18],
            s0=range_results[0], s1=range_results[1], s2=range_results[2], s3=range_results[3],
            s4=range_results[4], s5=range_results[5], s6=range_results[6], s7=range_results[7],
            s12=range_results[12], s13=range_results[13], s14=range_results[14], s15=range_results[15],
            s16=range_results[16], s17=range_results[17], s18=range_results[18])
    footer = \
"""\\botrule
\end{tabular}
\end{minipage}
%\end{center}
\end{table}"""
    print(header)
    print(middle_section)
    print(footer)



def show_file(tmp = files[0][0]):
    lines = read_file(tmp, path=path)
    for line in lines:
        print("{}, {}, {}, {}, {}".format(line[3], line[1], line[4], line[6], line[7]))


def print_item(row_idx, col_idx, lines):
    # Print headers
    print("{}, {}, {}, {}".format(lines[1][col_idx], lines[1][3], lines[1][4], lines[1][8]))
    print("{}, {}, {}, {}".format(lines[row_idx][col_idx], lines[row_idx][3],
                                  lines[row_idx][4], lines[row_idx][8]))


def print_all(col_idx = 1):
    print("{}, {}, {}, {}".format(lines[1][col_idx], lines[1][3], lines[1][4], lines[1][8]))
    for i, item_ in enumerate(lines):
        print("{}, {}, {}, {}".format(lines[i][col_idx], lines[i][3],
                                      lines[i][4], lines[i][8]))


def mean_and_std(dataset='hillstrom'):
    """
    Function for extracting mean and standard error for reruns of
    datasets 'hillstrom', 'starbucks', and 'voter'. The voter
    dataset is modified.
    """
    paths = [
        './DAMI3/results_default_seed/',
        './DAMI3/results_0_seed/',
        './DAMI3/results_1_seed/',
        './DAMI3/results_2_seed/',
        './DAMI3/results_3_seed/',
        # naive + dc + undersampling does not work for voter. Skip entirely.
        #'./DAMI3/results_4_seed/',
        './DAMI3/results_5_seed/',
        './DAMI3/results_6_seed/',
        './DAMI3/results_7_seed/',
        './DAMI3/results_8_seed/',
        './DAMI3/results_9_seed/']
    results = []
    for path in paths:
        results.append(parse_result_files(dataset, path))
    # results has n items
    # each item contains the total results of one run
    # each row in one item contains four text fields and
    # four numeric fields (first one might be list of two
    # items as it is two k-values). First one being optimal k,
    # second AUUC, third improvement to random, and fourth
    # EUCE.
    output = []
    for i, _ in enumerate(results[0]):
        # Get headers one model combination at a time
        output.append(results[0][i][:4])  # Add names
        # k-values
        # Check if item is float or string
        try:
            #print([item[i][4] for item in results])
            tmp_out = np.mean([float(item[i][4]) for item in results])
        except (ValueError, IndexError) as err:
            p = re.compile('[0-9]+.[0-9]+')
            k_values = []
            for item in results:
                tmp = p.findall(item[i][4])
                if len(tmp) < 2:
                    # Previous match did not produce expected results. Try differently:
                    q = re.compile('[0-9]+')  # Matches ints.
                    tmp = q.findall(item[i][4])
                k_values.append([float(tmp[0]), float(tmp[1])])
            # Average k-values:
            tmp_out = [np.mean([item[0] for item in k_values]), np.mean([item[1] for item in k_values])]
        output[i].append(tmp_out)
        # AUUC
        output[i].append(np.mean([float(item[i][5]) for item in results]))
        # "Improvement over random"
        output[i].append(np.mean([float(item[i][6]) for item in results]))
        # EUCE
        output[i].append(np.mean([float(item[i][7]) for item in results]))
    st_devs = []
    for i, _ in enumerate(results[0]):
        # Get headers one model combination at a time
        st_devs.append(results[0][i][:4])  # Add names
        # k-values
        try:
            tmp_out = np.std([float(item[i][4]) for item in results])
        except:
            p = re.compile('[0-9]+.[0-9]+')
            k_values = []
            for item in results:
                tmp = p.findall(item[i][4])
                if len(tmp) < 2:
                    # Previous match did not produce expected results. Try differently:
                    q = re.compile('[0-9]+')  # Matches ints.
                    tmp = q.findall(item[i][4])
                k_values.append([float(tmp[0]), float(tmp[1])])
            # Average k-values:
            tmp_out = [np.std([item[0] for item in k_values]), np.std([item[1] for item in k_values])]
        st_devs[i].append(tmp_out)
        # AUUC
        st_devs[i].append(np.std([float(item[i][5]) for item in results]))
        # "Improvement over random"
        st_devs[i].append(np.std([float(item[i][6]) for item in results]))
        # EUCE
        st_devs[i].append(np.std([float(item[i][7]) for item in results]))
    return output, st_devs


def median_and_range(dataset='hillstrom'):
    """
    Function for extracting median and range for reruns of
    datasets 'hillstrom', 'starbucks', and 'voter'. The voter
    dataset is modified.
    This is particularly for k-values.
    """
    paths = [
        './DAMI3/results_default_seed/',
        './DAMI3/results_0_seed/',
        './DAMI3/results_1_seed/',
        './DAMI3/results_2_seed/',
        './DAMI3/results_3_seed/',
        # naive + dc + undersampling does not work for voter. Skip entirely.
        #'./DAMI3/results_4_seed/',
        './DAMI3/results_5_seed/',
        './DAMI3/results_6_seed/',
        './DAMI3/results_7_seed/',
        './DAMI3/results_8_seed/',
        './DAMI3/results_9_seed/']
    results = []
    for path in paths:
        results.append(parse_result_files(dataset, path))
    # results has n items
    # each item contains the total results of one run
    # each row in one item contains four text fields and
    # four numeric fields (first one might be list of two
    # items as it is two k-values). First one being optimal k,
    # second AUUC, third improvement to random, and fourth
    # EUCE.
    output = []
    for i, _ in enumerate(results[0]):
        # Get headers one model combination at a time
        output.append(results[0][i][:4])  # Add names
        # k-values
        # Check if item is float or string
        try:
            #print([item[i][4] for item in results])
            tmp_out = np.median([float(item[i][4]) for item in results])
        except (ValueError, IndexError) as err:
            p = re.compile('[0-9]+.[0-9]+')
            k_values = []
            for item in results:
                tmp = p.findall(item[i][4])
                if len(tmp) < 2:
                    # Previous match did not produce expected results. Try differently:
                    q = re.compile('[0-9]+')  # Matches ints.
                    tmp = q.findall(item[i][4])
                k_values.append([float(tmp[0]), float(tmp[1])])
            # Average k-values:
            tmp_out = [np.median([item[0] for item in k_values]), np.median([item[1] for item in k_values])]
        output[i].append(tmp_out)
        # AUUC
        output[i].append(np.median([float(item[i][5]) for item in results]))
        # "Improvement over random"
        output[i].append(np.median([float(item[i][6]) for item in results]))
        # EUCE
        output[i].append(np.median([float(item[i][7]) for item in results]))
    st_devs = []
    for i, _ in enumerate(results[0]):
        # Get headers one model combination at a time
        st_devs.append(results[0][i][:4])  # Add names
        # k-values
        try:
            tmp_out = [np.min([float(item[i][4]) for item in results]),\
                np.max([float(item[i][4]) for item in results])]
        except:
            p = re.compile('[0-9]+.[0-9]+')
            k_values = []
            for item in results:
                tmp = p.findall(item[i][4])
                if len(tmp) < 2:
                    # Previous match did not produce expected results. Try differently:
                    q = re.compile('[0-9]+')  # Matches ints.
                    tmp = q.findall(item[i][4])
                k_values.append([float(tmp[0]), float(tmp[1])])
            # Average k-values:
            tmp_out = [[np.min([item[0] for item in k_values]), np.max([item[0] for item in k_values])],\
                [np.min([item[1] for item in k_values]), np.max([item[1] for item in k_values])]]
        st_devs[i].append(tmp_out)
        # AUUC
        st_devs[i].append([np.min([float(item[i][5]) for item in results]), np.max([float(item[i][5]) for item in results])])
        # "Improvement over random"
        st_devs[i].append([np.min([float(item[i][6]) for item in results]), np.max([float(item[i][6]) for item in results])])
        # EUCE
        st_devs[i].append([np.min([float(item[i][7]) for item in results]), np.max([float(item[i][7]) for item in results])])
    return output, st_devs


if __name__ == '__main__':
    # Run actual program:
    # Get results for criteo 1:
    criteo1 = parse_result_files('criteo1')

    # Get results for criteo 2:
    tmp = parse_result_files('criteo2')
    criteo2 = criteo2_hack(tmp)

    #Print latex-code:
    print("Latex code for criteo 1:")
    print("-" * 40)
    print("AUUC:")
    print_latex(criteo1)
    print("-" * 40)
    print("Optimal k:")
    print_latex(criteo1, 'k')
    print("-" * 40)
    print("EUCE")
    print_latex(criteo1, 'euce')
    print("=" * 40)
    print("Latex code for criteo 2:")
    print("-")
    print("AUUC:")
    print_latex(criteo2)
    print("-" * 40)
    print("Optimal k-values")
    print_latex(criteo2, 'k')
    print("-" * 40)
    print("EUCE:")
    print_latex(criteo2, 'euce')

    # Additional experiments after review 2:
    path = './DAMI3/results/'
    print("Modifying values for files[12] and files[13] for new datasets.")
    #files[12] = ['./../results_uplift_dc/results.csv', './../results_uplift_dc/results.csv_testing_set.csv']
    #files[13] = ['./../results_uplift_rf/results.csv', './../results_uplift_rf/results.csv_testing_set.csv']
    files[12] = ['./results_uplift_dc/results.csv', './results_uplift_dc/results.csv_testing_set.csv']
    files[13] = ['./results_uplift_rf/results.csv', './results_uplift_rf/results.csv_testing_set.csv']
    # hillstrom = parse_result_files('hillstrom')
    # print("\n")
    # print("Latex code for Hillstrom (conversion):")
    # print("-" * 40)
    # print("AUUC:")
    # print_latex(hillstrom)
    # print("-" * 40)
    # print("Optimal k:")
    # print_latex(hillstrom, 'k')
    # print("-" * 40)
    # print("EUCE")
    # print_latex(hillstrom, 'euce')
    hillstrom = mean_and_std('hillstrom')
    print("\n")
    print("Latex code for Hillstrom (conversion)")
    print_latex_w_std(hillstrom[0], hillstrom[1], 'auuc')
    print("-" * 40)
    print_latex_w_std(hillstrom[0], hillstrom[1], 'euce')
    print("-" * 40)
    #print_latex_w_std(hillstrom[0], hillstrom[1], 'k')
    hillstrom_k = median_and_range('hillstrom')
    print_latex_median_and_range(hillstrom_k[0], hillstrom_k[1])

    # starbucks = parse_result_files('starbucks')  # Missing naive DC
    # print("\n")
    # print("Latex code for Starbucks:")
    # print("-" * 40)
    # print("AUUC:")
    # print_latex(starbucks)
    # print("-" * 40)
    # print("Optimal k:")
    # print_latex(starbucks, 'k')
    # print("-" * 40)
    # print("EUCE")
    # print_latex(starbucks, 'euce')
    starbucks = mean_and_std('starbucks')
    print("\n")
    print("Latex code for Starbucks")
    print_latex_w_std(starbucks[0], starbucks[1], 'auuc')
    print("-" * 40)
    print_latex_w_std(starbucks[0], starbucks[1], 'euce')
    print("-" * 40)
    #print_latex_w_std(starbucks[0], starbucks[1], 'k')
    starbucks_k = median_and_range('starbucks')
    print_latex_median_and_range(starbucks_k[0], starbucks_k[1])

    # voter = parse_result_files('voter')  # Contains naive DC. Why are these three different?
    # print("\n")
    # print("Latex code for Voter (modified, 0.5\%):")
    # print("-" * 40)
    # print("AUUC:")
    # print_latex(voter)
    # print("-" * 40)
    # print("Optimal k:")
    # print_latex(voter, 'k')
    # print("-" * 40)
    # print("EUCE")
    # print_latex(voter, 'euce')
    voter = mean_and_std('voter')
    print("\n")
    print("Latex code for Voter (modified, p(y=1|t=1) = 0.5\%")
    print_latex_w_std(voter[0], voter[1], 'auuc')
    print("-" * 40)
    print_latex_w_std(voter[0], voter[1], 'euce')
    print("-" * 40)
    #print_latex_w_std(voter[0], voter[1], 'k')
    voter_k = median_and_range('voter')
    print_latex_median_and_range(voter_k[0], voter_k[1])

def s_p(std_1, std_2):
    return np.sqrt((std_1**2 + std_2**2) / 2)

def t(x_1, std_1, x_2, std_2, n_1, n_2):
    return (x_1 - x_2) / (s_p(std_1, std_2) * np.sqrt(1 / n_1 + 1 / n_2))