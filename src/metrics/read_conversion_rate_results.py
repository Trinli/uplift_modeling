"""
Code for reading in and plotting conversion rate experiment results.
"""


# Perhap we can reuse the 'find optimal results' -function
import src.metrics.read_dataset_size_results as dr
import experiments.conversion_rate_experiments as ce
from matplotlib import pyplot as plt

path = './results/conversion_rate/'
dr.path = path  # Set path for functions in dr

# 1. Generate file names for all result files

datasets = ce.datasets
models = ['dc_lr', 'uplift_rf']
undersampling_schemes = ['none', 'k_undersampling', 'split_undersampling']

results = []
simplified_results = []
for i, model in enumerate(models):
    for undersampling_scheme in undersampling_schemes:
        tmp_results = []
        tmp_euce = []
        for dataset in datasets:
            # Generate changing line for slurm
            # Generate file names.
            txt = './' + path
            txt += dataset + ' '
            txt += str(undersampling_scheme) + ' '
            txt += str(model) + ' '
            if undersampling_scheme == 'none':
                correction = 'none'
            elif undersampling_scheme == 'k_undersampling':
                correction = 'div_by_k'
            elif undersampling_scheme == 'split_undersampling':
                correction = 'analytic'
            txt += correction + ' '
            # Lastly result file
            # Name them with all details  # Not rate in ouput file name?
            output_file_name = dataset  # Write everything to result-directory
            output_file_name += '_' + undersampling_scheme
            output_file_name += '_' + model + '_' + correction
            output_file_name += '.csv'
            #csv_files.append(output_file_name)
            # Or perhaps read in the result here.
            try:
                res = dr.find_optimal_results(output_file_name)
                tmp_results.append(float(res[7]))
                tmp_euce.append(float(res[10]))
            except TypeError:
                tmp_results.append(None)
                tmp_euce.append(None)
            print(output_file_name)
            print(res)
            results.append({'result': res, 'file': output_file_name,
                            'model': model,
                            'dataset': dataset,
                            'undersampling_scheme': undersampling_scheme,
                            'correction': correction})
        simplified_results.append({'model': model, 
                                   'undersampling_scheme': undersampling_scheme,
                                   'results': tmp_results,
                                   'euce': tmp_euce})

# Order all model configurations' auuc into one line by decreasing dataset size
# One model configuration per line(!)

# 2. Create structure to save the results in
# 3. Read in the results
# 4. Plot or 
txt = [str(round(item, 4)) for item in ce.pt_rates]
tmp = [item['results'] for i, item in enumerate(simplified_results)]

labels = ['DC-LR', 'DC-LR (strat. und.)', 'DC-LR (split und.)',
          'Uplift RF', 'Uplift RF (strat. und.)', 'Uplift RF (split und.)']

linestyle = ['solid', 'dashed', 'dashdot', # 'dotted', # 
             'solid', 'dashed', 'dashdot'] #'dotted'] #
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
color = [colors[0], colors[0], colors[0],
         colors[2], colors[2], colors[2]]

# So apparently the corresponding best testing set result does not
# exist for DC-LR and 0.0078 and 0.0155. Exceeding memory requirements
# or time limit?
# And apparently the experiments produce very similar outputs,
# even identival.
# Standardize by dc-lr result!
relative_results = []
reference_line = simplified_results[0]['results']

def special_div(item_1, item_2):
    if item_1 == None:
        return None
    else:
        return item_1 / item_2 * 100

for line in simplified_results:
    tmp_res = [special_div(item_1, item_2) for item_1, item_2 in zip(line['results'], reference_line)]
    relative_results.append(tmp_res)

# Set figsize before plotting
resize = 0.8
plt.figure(figsize=(6.4 * resize, 4.8 * resize))
for i, line in enumerate(relative_results):
    #for i, line in enumerate(tmp):
    # One config at a t ime.
    #config = configurations[i + 10]  # + 10 to get to criteo 2 results.
    plt.plot(txt,
             #line['results'], 
             line,
             label=labels[i],
             linestyle=linestyle[i],
             color=color[i])
             #alpha=0.7)
    #plt.plot(txt_rates,
    #        [multiply(item) for item in config['results']],
    #        label=labels[i],
    #        linestyle=linestyle[i],
    #        color=color[i])

# Move legend out of plot
# ax.legend(loc='lower left', bbox_to_anchor=(1, .45))
# plt.subplots_adjust(bottom=.1, left=.15)

plt.legend(title="Model")  #, ncol=2)
#plt.yscale('log')
#plt.title("AUUC over dataset size")  # No title for plots in paper
plt.xlabel("p(y|t=1)")
plt.ylabel("% of DC-LR")
#fig, ax = plt.subplots()
#size = fig.get_size_inches()
#shrink = 0.8
#new_size = size * shrink
#fig(figsize=new_size)
#plt.rcParams["figure.figsize"] = (6.4 * shrink, 4.8 * shrink)
plt.savefig('conversion_rate.pdf', bbox_inches='tight')

#plt.show()


########################################################
########################################################
# Next generate latex code for table with same info
########################################################
########################################################


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
header += """\caption{mAUUC on Criteo-uplift 2}\label{some_table}
"""
# header +="""\\begin{tabular}{@{}llll@{}}
# \\toprule
#  $p_t$ & DC-LR & DC-LR (strat. und.) & DC-LR (split und.) \\\\
# \midrule"""


header +="""\\begin{tabular}{@{}llll@{}}
\\toprule
"""
header += """& {p_t[0]} & {p_t[1]} & {p_t[2]} & {p_t[3]} & {p_t[4]} & {p_t[5]} \\\\
\midrule""".format(p_t=txt)

middle_section = \
"""DC-LR & {r0[0]:.5f} & {r0[1]:.5f} & {r0[2]:.5f} & {r0[3]:.5f} & {r0[4]:.5f} & {r0[5]:.5f} \\\\
DC-LR (strat. und.) & {r1[0]:.5f} & {r1[1]:.5f} & {r1[2]:.5f} & {r1[3]:.5f} & {r1[4]:.5f} & {r1[5]:.5f} \\\\
DC-LR (split und.) & {r2[0]:.5f} & {r2[1]:.5f} & {r2[2]:.5f} & {r2[3]:.5f} & {r2[4]:.5f} & {r2[5]:.5f} \\\\
Uplift RF & {r3[0]:.5f} & {r3[1]:.5f} & {r3[2]:.5f} & {r3[3]:.5f} & {r3[4]:.5f} & {r3[5]:.5f} \\\\
Uplift RF (strat. und.) & {r4[0]:.5f} & {r4[1]:.5f} & {r4[2]:.5f} & {r4[3]:.5f} & {r4[4]:.5f} & {r4[5]:.5f} \\\\
Uplift RF (split und.) & {r5[0]:.5f} & {r5[1]:.5f} & {r5[2]:.5f} & {r5[3]:.5f} & {r5[4]:.5f} & {r5[5]:.5f} \\\\
""".format(p_t=labels, r0=simplified_results[0]['results'],
           r1=simplified_results[1]['results'],
           r2=simplified_results[2]['results'],
           r3=simplified_results[3]['results'],
           r4=simplified_results[4]['results'],
           r5=simplified_results[5]['results'])

footer = \
"""\\botrule
\end{tabular}
\end{minipage}
%\end{center}
\end{table}"""
print(header)
print(middle_section)
print(footer)
