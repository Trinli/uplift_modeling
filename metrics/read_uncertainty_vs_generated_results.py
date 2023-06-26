"""
Reading in results from tree_overall_theta_vs_ci.csv, i.e. experiments
where the predicted uncertainties were compared to generating parameters.

This is also done for the DGP-based model.
"""
import csv
import re
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14  # Increase default font size from 10 to 12.

# Read in file line by line:
rows = []
with open('./results/tree_overall_theta_vs_ci.csv', 'r') as handle:
    reader = csv.reader(handle, delimiter=';')
    headers = next(reader)
    for line in reader:
        rows.append(line)

# Parse list. AUUC is in index [7], all other interesting in index [2].
results = []
for line in rows:
    tmp = re.findall('\d+.\d+', line[2])
    tmp_theta_within_CI_CATE = float(tmp[0])
    tmp_theta_within_CI_ITE = float(tmp[1])
    tmp_average_CI = float(tmp[2])
    tmp = re.findall(' \d+', line[2])
    tmp_max_leaf_nodes = int(tmp[-2])
    tmp_min_leaf_size = int(tmp[-1])
    tmp_auuc = float(line[7])
    tmp = {'theta_CATE': tmp_theta_within_CI_CATE,
           'theta_ITE': tmp_theta_within_CI_ITE,
           'max_leaf_nodes': tmp_max_leaf_nodes,
           'min_leaf_size': tmp_min_leaf_size,
           'average_ci': tmp_average_CI,
           'AUUC': tmp_auuc}
    results.append(tmp)

# Populate matrix for visualization
results = sorted(results, key=lambda d: d['min_leaf_size'])
mat = [results[i*8:(i+1)*8] for i in range(9)]
mat = [sorted(item, key= lambda d: d['max_leaf_nodes']) for item in mat]

# Sanity check
for row in mat:
    print([item['max_leaf_nodes'] for item in row])
for row in mat:
    print([item['min_leaf_size'] for item in row])
theta_CATE = []
for row in mat:
    tmp = [item['theta_CATE'] for item in row]
    theta_CATE.append(tmp)
    print(tmp)
theta_ITE = []
for row in mat:
    tmp = [item['theta_ITE'] for item in row]
    theta_ITE.append(tmp)
    print(tmp)
AUUC = []  # Actually mAUUC.
for row in mat:
    tmp = [1000 * item['AUUC'] for item in row]
    AUUC.append(tmp)
    print(tmp)
average_CI = []
for row in mat:
    tmp = [item['average_ci'] for item in row]
    average_CI.append(tmp)
AUUC = np.array(AUUC)
average_CI = np.array(average_CI)
theta_CATE = np.array(theta_CATE)
theta_ITE = np.array(theta_ITE)

# MIGHT NEED TO RUN THE EXPERIMENT ALSO WITH LARGER MAX NUMBER OF LEAF NODES.

plt.rcParams['font.size'] = 12
fig, ax = plt.subplots(2, 2, sharey=True, sharex=True)
fig.tight_layout(pad=4)

# AUUC
im_aci = ax[0, 0].imshow(AUUC, cmap='plasma_r')
plt.colorbar(im_aci, ax=ax[0, 0], fraction=0.05)
# Add ticks and something.
ax[0, 0].set_yticks([i for i in range(9)], [str(2**(item+4)) for item in range(9)])
#ax[0, 0].set_xticks([i for i in range(8)], [r'$2^1$', r'$2^2$', r'$2^3$', r'$2^4$', r'$2^5$', r'$2^6$', r'$2^7$', r'$2^8$'])
ax[0, 0].set_title(r"mAUUC")
#plt.ylabel('Min. samples in node')
#plt.xlabel('Max. number of leaf nodes')
ax[0, 0].set_ylabel('Min. samples in node')
#ax[0, 0].set_xlabel('Max. number of leaf nodes')

# Average CI
im_aci = ax[0, 1].imshow(average_CI, cmap='viridis')
plt.colorbar(im_aci, ax=ax[0, 1], fraction=0.05)
# Add ticks and something.
#ax[0, 1].set_yticks([i for i in range(9)], [str(2**(item+4)) for item in range(9)])
#ax[0, 1].set_xticks([i for i in range(8)], [r'$2^1$', r'$2^2$', r'$2^3$', r'$2^4$', r'$2^5$', r'$2^6$', r'$2^7$', r'$2^8$'])
ax[0, 1].set_title("Average CI")
#plt.ylabel('Min. samples in node')
#plt.xlabel('Max. number of leaf nodes')
#ax[0, 1].set_ylabel('Min. samples in node')
#ax[0, 1].set_xlabel('Max. number of leaf nodes')

# theta CATE
im_aci = ax[1, 0].imshow(theta_CATE, cmap='plasma_r')
plt.colorbar(im_aci, ax=ax[1, 0], fraction=0.05)
# Add ticks and something.
ax[1, 0].set_yticks([i for i in range(9)], [str(2**(item+4)) for item in range(9)])
ax[1, 0].set_xticks([i for i in range(8)], [r'$2^1$', r'$2^2$', r'$2^3$', r'$2^4$', r'$2^5$', r'$2^6$', r'$2^7$', r'$2^8$'])
ax[1, 0].set_title(r"$\bar{\theta}_i$ within CI (CATE)")
#plt.ylabel('Min. samples in node')
#plt.xlabel('Max. number of leaf nodes')
ax[1, 0].set_ylabel('Min. samples in node')
ax[1, 0].set_xlabel('Max. number of leaf nodes')

# theta ITE
im_aci = ax[1, 1].imshow(theta_ITE, cmap='viridis')
plt.colorbar(im_aci, ax=ax[1, 1], fraction=0.05)
# Add ticks and something.
#ax[1, 1].set_yticks([i for i in range(9)], [str(2**(item+4)) for item in range(9)])
ax[1, 1].set_xticks([i for i in range(8)], [r'$2^1$', r'$2^2$', r'$2^3$', r'$2^4$', r'$2^5$', r'$2^6$', r'$2^7$', r'$2^8$'])
ax[1, 1].set_title(r"$\theta_i$ within CI (ITE)")
#plt.ylabel('Min. samples in node')
#plt.xlabel('Max. number of leaf nodes')
#ax[1, 1].set_ylabel('Min. samples in node')
ax[1, 1].set_xlabel('Max. number of leaf nodes')

plt.savefig('./results/uncertainty/tree_theta_within_ci.pdf')
plt.clf()
