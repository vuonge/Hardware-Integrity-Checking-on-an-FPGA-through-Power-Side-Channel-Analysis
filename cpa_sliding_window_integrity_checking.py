import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, shapiro, mannwhitneyu, pearsonr
import sys
import argparse

# -----------------------------
# ARG PARSING
# -----------------------------

parser = argparse.ArgumentParser(description="Pearson histogram experiment")

parser.add_argument("--groupA", required=True, help="Group A dataset file")
parser.add_argument("--groupB", required=True, help="Group B dataset file")

args = parser.parse_args()

GROUP_A_FILE = args.groupA
GROUP_B_FILE = args.groupB

# -----------------------------
# USER-TUNABLE PARAMETERS, CHANGE HERE
# -----------------------------
ALPHA = 0.000001   # Probability of False Positive (FP)

WINDOW_SIZE = 256
NUM_RUNS = 50 # doesn't check if the number of runs will end up exceeding the file length
SAMPLES_PER_TRACE = 256

NUM_TRACES_KNOWN = 300
NUM_TRACES_TEST = 5

GOLDEN_TRACE_IDX = 0 # Which trace (of Group A) to use as the golden template
GOLDEN_WINDOW_IDX = 0 # Which window of the golden trace to use as the template

# -----------------------------
# DATASETS, CHANGE HERE
# -----------------------------
# GROUP_A_FILE = "basys3/sw/bin/aes_std_2_15_1/sensor_traces_5k.csv"
# GROUP_B_FILE = "basys3/sw/bin/aes_std_2_14_1/sensor_traces_5k.csv"


# -----------------------------
# DATA LOADING FUNCTIONS
# -----------------------------

# Returns 1D array of samples as hamming weights
def load_traces(filename):
    points = []
    with open(filename, "r") as f:
        line = f.readline() # a trace
        while line:
            trace = line.strip().split(",")
            
            hws = []
            for sample in trace:
                hw = [bin(int(sample, 16)).count("1")][0]
                hws.append(hw)
                
            points.extend(np.array(hws))
            
            line = f.readline()
    return points

log_name = f"log_{GROUP_A_FILE.split('/')[-2]}_vs_{GROUP_B_FILE.split('/')[-2]}.txt"
sys.stdout = open(log_name, "w")

# Group A is benign. Used for both the golden template and the "known" distribution.
# Group B can be benign or malicious. It used for the "test" distribution.

hw_group_a = load_traces(GROUP_A_FILE)

# Golden template is the first 256 points (first trace)
golden_template = hw_group_a[GOLDEN_TRACE_IDX:(GOLDEN_TRACE_IDX+256)]
hw_group_a = hw_group_a[256*1:256*(NUM_TRACES_KNOWN+1)] # known, skip golden template, 300 * 256 = 76800 points
# print(golden_template)
# print(len(hw_group_a))
hw_group_b_orig = load_traces(GROUP_B_FILE) # test, skip golden template, only a few (5*256 = 1280) points
hw_group_b = hw_group_b_orig

reject_count = 0
fail_count = 0
for j in range(NUM_RUNS):
    # For each run, change where the test starting trace is
    hw_group_b = hw_group_b_orig[256*(j):]

    p_dist_known_all = [] # Pearson distribution for known across runs
    p_dist_test_all = []  # Pearson distribution for test across runs

    for i in range((NUM_TRACES_KNOWN-1)*256): # -1 because need 256 room for last window
        known_window = hw_group_a[i : i+256] 
        r, p_value = pearsonr(golden_template, known_window)
        p_dist_known_all.append(r*r) # using r squared
        
    for i in range((NUM_TRACES_TEST-1)*256):
        test_window = hw_group_b[i : i+256]
        r, p_value = pearsonr(golden_template, test_window)
        p_dist_test_all.append(r*r)
                
        
    # -------
    p_dist_known = np.array(p_dist_known_all)
    p_dist_test = np.array(p_dist_test_all)


    # # Optional: Shapiro-Wilk normality check
    shapiro_known_stat, shapiro_known_p = shapiro(p_dist_known)
    shapiro_test_stat, shapiro_test_p = shapiro(p_dist_test)
    print(f"KNOWN: Shapiro stat: {shapiro_known_stat}, p: {shapiro_known_p}")
    print(f"TEST: Shapiro stat: {shapiro_test_stat}, p: {shapiro_test_p}")

    print(f"RUN {j}:")
    u_stat, p_val = mannwhitneyu(p_dist_known, p_dist_test, alternative='two-sided')
    print(f"\nU-statistic: {u_stat:.3f}")
    print(f"P-value: {p_val}")
    print("-------------------------------------")

    reject_null = p_val < ALPHA

    if reject_null:
        reject_count += 1
    else:
        fail_count += 1
        

    

# -----------------------------
# HISTOGRAM VISUALIZATION
# -----------------------------

min_p = min(p_dist_known.min(), p_dist_test.min())
max_p = max(p_dist_known.max(), p_dist_test.max())

NUM_BINS = 100   # tweak this if needed
bins = np.linspace(min_p, max_p, NUM_BINS)

mean_p_known = np.mean(p_dist_known)
std_p_known  = np.std(p_dist_known)

mean_p_test = np.mean(p_dist_test)
std_p_test  = np.std(p_dist_test)

plt.figure(figsize=(10, 4))

# ---- Known ----
plt.subplot(1, 2, 1)
plt.hist(p_dist_known, bins=bins, alpha=0.75, edgecolor='black')
plt.axvline(mean_p_known, linestyle='-', linewidth=2, label=f"Mean = {mean_p_known:.3f}")
plt.axvline(mean_p_known + std_p_known, linestyle='--', linewidth=1, label=f"+1σ = {std_p_known:.3f}")
plt.axvline(mean_p_known - std_p_known, linestyle='--', linewidth=1)
plt.xlabel("Pearson Correlation Coefficient Squared (r^2)")
plt.ylabel("Occurrences")
plt.title("Known Pearson r^2 Histogram")
plt.legend()
plt.grid(True)

# ---- Test ----
plt.subplot(1, 2, 2)
plt.hist(p_dist_test, bins=bins, alpha=0.75, edgecolor='black')
plt.axvline(mean_p_test, linestyle='-', linewidth=2, label=f"Mean = {mean_p_test:.3f}")
plt.axvline(mean_p_test + std_p_test, linestyle='--', linewidth=1, label=f"+1σ = {std_p_test:.3f}")
plt.axvline(mean_p_test - std_p_test, linestyle='--', linewidth=1)
plt.xlabel("Pearson Correlation Coefficient Squared (r^2)")
plt.ylabel("Occurrences")
plt.title("Test Pearson r^2 Histogram")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"pearson_hist_{GROUP_A_FILE.split('/')[-2]}_vs_{GROUP_B_FILE.split('/')[-2]}.pdf", format="pdf", bbox_inches="tight")


# -----------------------------
# SUMMARY
# -----------------------------
print("\n===== EXPERIMENTAL RESULTS =====")
print(f"GROUP A: {GROUP_A_FILE}")
print(f"GROUP B: {GROUP_B_FILE}")
print(f"ALPHA USED (False Positive Rate): {ALPHA}")
print(f"Number of trials: {NUM_RUNS}")
print(f"Null hypothesis rejected: {reject_count} times")
print(f"Null hypothesis NOT rejected: {fail_count} times")

# plt.show()

