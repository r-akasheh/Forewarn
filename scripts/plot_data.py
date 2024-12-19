import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO


# Load the data into a DataFrame
data = pd.read_csv('../vlm/llama-recipes/no_hist_results_update.csv')
data2 = pd.read_csv('../vlm/llama-recipes/hist_results_update.csv')
data = pd.concat([data, data2])
# Total counts for train and test
total_failures_train = 172
total_failures_test = 17
total_successes_train = 128
total_successes_test = 19

# Convert rates to absolute values
def convert_to_absolute(row):
    if row['group'] == 'train':
        row['TP'] *= total_successes_train
        row['FN'] *= total_successes_train
        row['TN'] *= total_failures_train
        row['FP'] *= total_failures_train
    else:  # Test group
        row['TP'] *= total_successes_test
        row['FN'] *= total_successes_test
        row['TN'] *= total_failures_test
        row['FP'] *= total_failures_test
    return row

data = data.apply(convert_to_absolute, axis=1)

# Calculate TPR, TNR, FPR, FNR, Precision, and Recall
data['TPR'] = data.apply(
    lambda row: row['TP'] / (total_successes_train if row['group'] == 'train' else total_successes_test), axis=1
)
data['TNR'] = data.apply(
    lambda row: row['TN'] / (total_failures_train if row['group'] == 'train' else total_failures_test), axis=1
)
data['FPR'] = data.apply(
    lambda row: row['FP'] / (total_failures_train if row['group'] == 'train' else total_failures_test), axis=1
)
data['FNR'] = data.apply(
    lambda row: row['FN'] / (total_successes_train if row['group'] == 'train' else total_successes_test), axis=1
)
data['Precision'] = data['TP'] / (data['TP'] + data['FP'])
data['Recall'] = data['TPR']  # Recall is the same as TPR
data['Accuracy'] = (data['TP'] + data['TN']) / (data['TP'] + data['TN'] + data['FP'] + data['FN'])

# Define the metrics to plot
metrics = ["TPR", "TNR", "FPR", "FNR", "Precision", "Recall", 'Accuracy']
metric_labels = ["True Positive Rate (TPR)", "True Negative Rate (TNR)", 
                 "False Positive Rate (FPR)", "False Negative Rate (FNR)", 
                 "Precision", "Recall", "Accuracy"]
curve_types = ["no_hist", "hist"]

# Plotting
for metric, label in zip(metrics, metric_labels):
    plt.figure(figsize=(12, 6))
    for group in ["train", "test"]:
        group_data = data[data['group'] == group]
        for curve_type in curve_types:
            # Filter the data for each curve type
            filtered_data = group_data[group_data['type'] == curve_type]
            plt.plot(filtered_data['start_index'], filtered_data[metric], 
                     label=f"{group.capitalize()} - {curve_type}")
            
            # Annotate values on the plot
            for _, row in filtered_data.iterrows():
                plt.text(row['start_index'], row[metric], 
                         f"{row[metric]:.2f}", fontsize=8, 
                         ha='center', va='bottom')
    
    plt.xlabel("Start Index")
    plt.ylabel(label)
    plt.title(f"{label} vs Start Index (Train and Test Data)")
    plt.legend(title="Group and Type")
    plt.grid(True)
    plt.show()
