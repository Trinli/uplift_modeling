"""
This file contains code to reproduce the graph for explaining AUUC used in my thesis.
Run from terminal with "python -m metrics.auuc_visualization"
"""

import metrics.uplift_metrics as um
import data.load_data as ld
import models.double_classifier as dc
import matplotlib.pyplot as plt
import numpy as np


def visualize_auuc():
    """
    Function for visualizing AUUC. The point is
    to explain what the curve actually is.
    """
    # Let's use the starbucks data
    data = ld.get_starbucks_data()

    # Train some model
    model = dc.LogisticRegression()
    model.fit(data)
    # Predict
    pred = model.predict_uplift(data['testing_set']['X'])
    # Get some uplift points. These points are the expected conversion rates:
    conv = um._expected_conversion_rates(data['testing_set']['y'], pred, data['testing_set']['t'])
    # Note that tmp[0] contains the expected conversion rate for untreated observations and
    # tmp[-1] the expected conversion rate for treated observations.
    conv_c = conv[0]
    conv_t = conv[-1]
    n_observations = len(conv)
    fig = plt.figure(figsize=(16, 6))
    #plt.rcParams['figure.figsize'] = 1280, 480
    #ax11 = fig.add_subplot(121)
    ax12 = fig.add_subplot(121)
    ax21 = fig.add_subplot(122)
    plt.rcParams['font.size'] = 28
    x_points = [item / (n_observations - 1) for item in range(n_observations)]
    # 1. Plot "regular" uplift curve, i.e. expected conversion rate - expected conversion
    # rate for random treatment plan for the same treatment rate
    def interpolate(i, n_observations = n_observations, conv_c=conv_c, conv_t=conv_t):
        return i / n_observations * conv_t + (n_observations - i) / n_observations * conv_c

    y_points_1 = [conv_item - interpolate(i) for i, conv_item in enumerate(conv)]
    zeros = [0 for _ in x_points]
    # No one actually uses this form:
    # ax11.fill_between(x_points, 0, y_points_1, color='tab:blue', alpha=0.3)
    # ax11.plot(x_points, y_points_1, color='tab:blue')
    # ax11.plot(x_points, zeros, color='tab:orange')
    # #ax11.set_ylabel(r'$\mathbf{E}(\tau)$')
    # ax11.set_ylabel('Uplift')
    # 2. Plot uplift curve (our version) where the points are the expected conversion
    # rate subtracted by conv_c. Zero the graph at conv_c. Highlight the random area.
    y_points_3 = conv
    y_points_rand = [interpolate(i) for i, _ in enumerate(conv)]
    y_points_2 = [item - conv_c for item in conv]
    y_points_rand_2 = [item - conv_c for item in y_points_rand]
    tmp_conv = [conv_c for _ in x_points]
    ax21.fill_between(x_points, y_points_rand_2, y_points_2, color='tab:blue', alpha=0.3)  # Area between curve and rand
    ax21.plot(x_points, y_points_2, color='tab:blue')
    ax21.fill_between(x_points, 0, y_points_rand_2, color='tab:green', alpha=0.3)  # Area of rand
    ax21.plot(x_points, y_points_rand_2, color='tab:orange')
    #ax21.set_ylabel(r'$\mathbf{E}(y)$')
    ax21.set_ylabel('Uplift')
    ax21.set_xlabel('Treatment rate')
    # 3. Plot the expected conversion rate. Graph zeroed at zero. Add suitable shapes
    # to illustrate the random area, the area under the expected conversion rates
    ax12.fill_between(x_points, y_points_rand, y_points_3, color='tab:blue', alpha=0.3)  # AUUC area
    ax12.plot(x_points, y_points_3, color='tab:blue')  # AUUC curve
    #ax12.fill_between(x_points, 0, y_points_3, hatch='\\', alpha=0.0)  # E(r) area
    
    #ax12.fill_between(x_points, 0, tmp_conv, hatch='x', alpha=0.0)  # "New" random area
    #ax12.fill_between(x_points, 0, y_points_rand, color='tab:green', alpha=0.3)  # Random area
    ax12.fill_between(x_points, 0, tmp_conv, color='tab:red', alpha=0.3)  # "New" random area
    ax12.fill_between(x_points, tmp_conv, y_points_rand, color='tab:green', alpha=0.3)  # Random area
    
    ax12.plot(x_points, y_points_rand, color='tab:orange')  # Random line
    #ax12.fill_between(x_points, 0, y_points_rand, hatch='/', alpha=0.0)  # Random area
    ax12.set_ylim((0, max(y_points_3)*1.02))  # Add 2% gap
    ax12.set_ylabel(r'$\mathbb{E}(y)$')
    ax12.set_xlabel('Treatment rate')
    #fig.supxlabel('Treatment rate')
    # Don't set y-range. Sets smallest to min(y_points) automatically.
    plt.tight_layout(h_pad=1)
    plt.savefig('./figures/auuc.pdf')  #, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    # Run actual code
    visualize_auuc()
