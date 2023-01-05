"""
This file can read results from grid-search experiments and plot them into heatmaps.
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import re
from itertools import product


def compile_results_from_csv(result_file='./results_uplift_rf/grid_search/results.csv',
                             metric_idx=7,
                             output_file='./results_uplift_rf/grid_search/heatmap.pdf',
                             plot=True):
    """
    Function for compiling results from csv-file to matrix?

    Args:
    result_file (str): Location of csv file with results.
    metric_idx (int): Index of metric of interest. 7 is AUUC,
     10 is EUCE, 11 is MUCE.
    plot (bool): If True, will produce a heatmap. Otherwise will only
     return the data matrix.
    """
    # Below, or extract dimensionality from csv?
    # Open result file, decide from there what k's
    # were used.
    csv_lines = []
    with open(result_file, 'r') as handle:
        # Parse file content into list
        # to be populated into matrix later.
        # Super simple as the file is a csv-file :)
        reader = csv.reader(handle, delimiter=';')
        # Skip first row (headers)
        names = reader.__next__()
        for line in reader:
            csv_lines.append(line)
    tmp = []
    for item in csv_lines:
        # Regular expression to extract k_t and k_c values
        numbers = re.findall('\d+', item[4])  # item[1] for original results.
        k_t_tmp = float(numbers[0] + '.' + numbers[1]) #float(numbers[3] + '.' + numbers[4])  # String concatenation
        k_c_tmp = float(numbers[2] + '.' + numbers[3])  # float(numbers[5] + '.' + numbers[6])
        if k_t_tmp == 168.2694283879255:  # Drop the lone 1:1 sample.
            continue
        tmp.append([float(k_t_tmp), float(k_c_tmp), float(item[metric_idx])])  # Item 7 is 'improvement over random'?
    # Arrange into matrix:
    # Reverse order for better plot:
    rows = np.unique([item[0] for item in tmp])[::-1]
    cols = np.unique([item[1] for item in tmp])
    # Populate a matrix
    mat = np.empty(shape=(len(rows), len(cols)), dtype=np.float32)
    mat[:, :] = np.nan
    for item in tmp:
        # Match k_t and k_c to positions in rows and cols
        idx_t = np.where(rows == item[0])
        idx_c = np.where(cols == item[1])
        # Place item[2] in that position
        mat[idx_t, idx_c] = item[2]

    if plot:
        # Plot heatmap of metrics
        # This plotting part is full of customs stuff for specifically
        # the Uplift RF plot.
        fig, ax = plt.subplots()
        fig.canvas.draw()
        x_labels = [item.get_text() for item in ax.get_xticklabels()]
        x_labels = [str(0)] + [str(item) for item in cols[::2]]
        ax.set_xticklabels(x_labels)
        y_labels = [item.get_text() for item in ax.get_yticklabels()]
        y_labels = [str(0)] + [str(item) for item in rows[::2]]
        ax.set_yticklabels(y_labels)
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        ax.set_xlabel('$k_c$')
        ax.set_ylabel('$k_t$')
        #plt.xlabel('k_c')
        #plt.ylabel('k_t')
        plt.title('AUUC for Uplift Random Forest')
        #plt.title('AUUC for DC-LR')
        # heatmap_ = plt.get_cmap()
        heatmap_ = plt.get_cmap('hot')  # 'hot_r' for reversed colors
        heatmap_.set_bad('darkblue')
        # We might want to set the color range to get comparable
        # colors for all experiments
        plt.imshow(mat, cmap=heatmap_)
        plt.colorbar(shrink=.535)
        # Mark the diagonal where k_t == k_c:
        plt.scatter(x=[0, 2, 4, 6, 8, 10, 12,  14, 16],
                    y=[8, 7, 6, 5, 4, 3, 2, 1, 0],
                    marker='.', c='black')
        # Mark the pseud-diagonal where p_t == p_c:
        plt.scatter(x=[1, 3, 5, 7, 9, 11, 13, 15, 17],
                    y=[8, 7, 6, 5, 4, 3, 2, 1, 0],
                    marker='+', c='black')
        plt.scatter(x=[0], y=[8], marker='o', c='black')
        plt.savefig(output_file, bbox_inches='tight')

    # Outputs could be used for separate plotting function
    return {'csv': tmp, 'headers': names, 'data': mat, 'cols': cols, 'rows': rows}


def generate_heatmaps():
    """
    Function for generating heatmaps of both DC-LR and Uplift RF
    with split undersampling.
    """
    rf_result_file = './results/criteo2_uplift_rf_grid_search.csv'
    dc_result_file = './results/criteo2_dc_lr_grid_search.csv'
    metric_idx = 7
    output_file = './results/heatmap.pdf'

    rf_data = compile_results_from_csv(rf_result_file, plot=False)
    dc_data = compile_results_from_csv(dc_result_file, plot=False)
    # Change unit from AUUC to mAUUC for better legibility:
    rf_data['data'] = 1000 * rf_data['data']
    dc_data['data'] = 1000 * dc_data['data']    
    # Find range of values
    rf_min, rf_max = np.nanmin(rf_data['data']), np.nanmax(rf_data['data'],)
    dc_min, dc_max = np.nanmin(dc_data['data']), np.nanmax(dc_data['data'])
    _min = min(rf_min, dc_min)
    _max = max(rf_max, dc_max)

    # RF Max idx:
    rf_max_idx = np.unravel_index(np.nanargmax(rf_data['data'], axis=None), rf_data['data'].shape)
    dc_max_idx = np.unravel_index(np.nanargmax(dc_data['data'], axis=None), dc_data['data'].shape)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.canvas.draw()
    ax1.set_title(label='DC-LR')  # Should be a different command for title
    ax2.set_title(label='Uplift RF')  # Should be a different command for title    

    for ax, data in zip([ax1, ax2], [dc_data, rf_data]):
        mat = data['data']
        x_labels = [str(item) for item in data['cols']]
        ax.set_xticks(ticks=[0, 4, 8, 12, 16])  # Set location of ticks
        ax.set_xticklabels(labels=['1', '4', '16', '64', '256'])  # Set values of ticks
        ax.set_yticks(ticks=[8, 6, 4, 2, 0]) #[1, 3, 5, 7, 9])
        ax.set_yticklabels(labels=['1', '4', '16', '64', '256'])
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        ax.set_xlabel('$k_{t=0}$')
        ax.set_ylabel('$k_{t=1}$')
        # heatmap_ = plt.get_cmap()
        heatmap_ = plt.get_cmap('hot')  # 'hot_r' for reversed colors
        heatmap_.set_bad('darkblue')
        # We might want to set the color range to get comparable
        # colors for all experiments
        fn = ax.imshow(mat, cmap=heatmap_, vmin=_min, vmax=_max)
        # Mark the diagonal where k_t == k_c:
        ax.scatter(x=[0, 2, 4, 6, 8, 10, 12,  14, 16],
                    y=[8, 7, 6, 5, 4, 3, 2, 1, 0],
                    marker='.', c='black')
        # Mark the pseudo-diagonal where p_t == p_c:
        ax.scatter(x=[1, 3, 5, 7, 9, 11, 13, 15, 17],
                    y=[8, 7, 6, 5, 4, 3, 2, 1, 0],
                    marker='+', c='black')
        ax.scatter(x=[0], y=[8], marker='x', c='black')
        max_idx = np.unravel_index(np.nanargmax(mat, axis=None), mat.shape)
        ax.scatter(x=max_idx[1], y=max_idx[0], marker='*', c='black')
        ax.label_outer()
    tp = fig.colorbar(fn, ax=(ax1, ax2), shrink=.6, location='bottom')
    tp.set_label('mAUUC')
    #plt.show()
    plt.savefig(output_file, bbox_inches='tight')


def plot_undersampling_plot():
    """
    Function for plotting the undersampling figure needed for the paper.
    Two images, one with like 1% "positive" (red?) samples (dots?), another
    one where half of the negative (blue?) are dropped.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)
    n_x = 20
    n_y = 30
    x_tmp = [i for i in range(n_x)]
    y_tmp = [i for i in range(n_y)]
    points = product(x_tmp, y_tmp)
    x = [item[0] for item in points]
    points = product(x_tmp, y_tmp)
    y = [item[1] for item in points]
    # Randomly select 1% of the "observations" to be positive (different color)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    def randomize_color():
        if random.uniform(0, 1) > 0.99:
            return colors[1]
        else:
            return colors[0]
    color_list = [randomize_color() for item in y]
    def drop_points(item):
        if item == colors[1]:
            return item
        elif random.uniform(0, 1) > 0.5:
            return '#ffffff' # White color?
        else:
            return item

    ax1.scatter(x, y, c=color_list)  # c - list of colors
    ax1.axis('off')

    colors_dropped = [drop_points(item) for item in color_list]
    ax2.scatter(x, y, c=colors_dropped)
    ax2.axis('off')

    # Add rectangle
    rect = patches.Rectangle((7.5, 10.5), 10, 10, facecolor='none', edgecolor='r')
    ax1.add_patch(rect)
    rect_2 = patches.Rectangle((7.5, 10.5), 10, 10, facecolor='none', edgecolor='r')
    ax2.add_patch(rect_2)
    plt.savefig("undersampling.pdf", bbox_inches='tight')
    #plt.show()
