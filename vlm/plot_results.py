import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Define the experiment data
experiments = {
    # "VLM Before Finetuning (Eval)": {"class_acc_fail": 0.9259, "class_acc_success": 0.1481, "class_acc_fnr": 0.8519, "class_acc_fpr": 0.0741, "overall_acc": 0.5370},
    # "VLM Before Finetuning (Train)": {"class_acc_fail": 0.8700, "class_acc_success": 0.2100, "class_acc_fnr": 0.7900, "class_acc_fpr": 0.1300, "overall_acc": 0.5400},
    # "VLM After Finetuning (Eval)": {"class_acc_fail": 0.9259, "class_acc_success": 0.8889, "class_acc_fnr": 0.1111, "class_acc_fpr": 0.0741, "overall_acc": 0.9074},
    # "VLM After Finetuning (Train)": {"class_acc_fail": 0.9500, "class_acc_success": 0.9600, "class_acc_fnr": 0.0400, "class_acc_fpr": 0.0500, "overall_acc": 0.9550},
    # "VLM After Finetuning (Reduced Params, Eval)": {"class_acc_fail": 0.8889, "class_acc_success": 0.8148, "class_acc_fnr": 0.1852, "class_acc_fpr": 0.1111, "overall_acc": 0.8519},
    # "VLM After Finetuning (Reduced Params, Train)": {"class_acc_fail": 0.8889, "class_acc_success": 0.8889, "class_acc_fnr": 0.1111, "class_acc_fpr": 0.1111, "overall_acc": 0.8889},
    # "VLM + WM Before Finetuning (Train)": {"class_acc_fail": 0.8300, "class_acc_success": 0.1400, "class_acc_fnr": 0.8600, "class_acc_fpr": 0.1700, "overall_acc": 0.4850},
    # "VLM + WM Before Finetuning (Eval)": {"class_acc_fail": 0.8148, "class_acc_success": 0.0741, "class_acc_fnr": 0.9259, "class_acc_fpr": 0.1852, "overall_acc": 0.4444},
    # "VLM + WM After Finetuning (Train)": {"class_acc_fail": 0.9100, "class_acc_success": 0.9400, "class_acc_fnr": 0.0600, "class_acc_fpr": 0.0900, "overall_acc": 0.9250},
    # "VLM + WM After Finetuning (Eval)": {"class_acc_fail": 0.8519, "class_acc_success": 0.8519, "class_acc_fnr": 0.1481, "class_acc_fpr": 0.1481, "overall_acc": 0.8519},
    # Add more experiments as needed
    "Llama Image (Train)": {"class_acc_fail": 0.8800, "class_acc_success": 0.1200, "class_acc_fnr": 0.1200, "class_acc_fpr": 0.8800, "overall_acc": 0.5000},
    "Llama Image (Eval)":{"class_acc_fail": 0.8148, "class_acc_success": 0.0741, "class_acc_fnr": 0.1852, "class_acc_fpr": 0.9259, "overall_acc": 0.4444},
    "LLama Image Finetuned (Train)": {"class_acc_fail": 0.9800, "class_acc_success": 0.9600, "class_acc_fnr": 0.0200, "class_acc_fpr": 0.0400, "overall_acc": 0.9700},
    "LLama Image Finetuned (Eval)": {"class_acc_fail": 0.9259, "class_acc_success": 0.8889, "class_acc_fnr": 0.0741, "class_acc_fpr": 0.1111, "overall_acc": 0.9074},
    "Llama Latent GT (Train)": {"class_acc_fail": 0.9900, "class_acc_success": 0.0, "class_acc_fnr": 1.0, "class_acc_fpr": 0.0100, "overall_acc": 0.4950},
    "Llama Latent GT (Eval)": {"class_acc_fail": 0.9630, "class_acc_success": 0.1111, "class_acc_fnr": 0.8889, "class_acc_fpr": 0.0370, "overall_acc": 0.5370},
    "Llama Latent GT Finetuned (Train)": {"class_acc_fail": 0.9200, "class_acc_success": 0.9500, "class_acc_fnr": 0.0500, "class_acc_fpr": 0.0800, "overall_acc": 0.9350},
    "Llama Latent GT Finetuned (Eval)": {"class_acc_fail": 0.8148, "class_acc_success": 0.9259, "class_acc_fnr": 0.0741, "class_acc_fpr": 0.1852, "overall_acc": 0.8704},
    "Llama Latent Imagined (Train)": {"class_acc_fail": 0.9800, "class_acc_success": 0.4900, "class_acc_fnr": 0.5100, "class_acc_fpr": 0.0200, "overall_acc": 0.7350},
    "Llama Latent Imagined (Eval)":  {"class_acc_fail": 1.0, "class_acc_success": 0.2593, "class_acc_fnr": 0.7407, "class_acc_fpr": 0.0, "overall_acc": 0.6296},
    "Llama Latent Imagined Finetuned (Train)": {"class_acc_fail": 0.9400, "class_acc_success": 0.9700, "class_acc_fnr": 0.0300, "class_acc_fpr": 0.0600, "overall_acc": 0.9550},
    "Llama Latent Imagined Finetuned (Eval)": {"class_acc_fail": 0.8148, "class_acc_success": 0.8148, "class_acc_fnr": 0.1852, "class_acc_fpr": 0.1852, "overall_acc": 0.8148},
    "Llama Latent GT Snippet Finetuned (Train)": {"class_acc_fail": 0.7907, "class_acc_success": 0.9836, "class_acc_fnr": 0.0164, "class_acc_fpr": 0.2093, "overall_acc": 0.8877},
    "Llama Latent GT Snippet Finetuned (Eval)":  {"class_acc_fail": 0.6631, "class_acc_success": 0.9320, "class_acc_fnr": 0.0680, "class_acc_fpr": 0.3369, "overall_acc": 0.8126},
    "Llama Latent Imagined Snippet Finetuned (Train)": {"class_acc_fail": 0.7618, "class_acc_success": 0.9918, "class_acc_fnr": 0.0082, "class_acc_fpr": 0.2382, "overall_acc": 0.8774},
    "Llama Latent Imagined Snippet Finetuned (Eval)": {"class_acc_fail": 0.6702, "class_acc_success": 0.9603, "class_acc_fnr": 0.0397, "class_acc_fpr": 0.3298, "overall_acc": 0.8315},
}
# Create a DataFrame from the experiment data
data = pd.DataFrame(experiments).T
# Define color scheme for train/eval pairs
distinct_colors = [
    ('#1f77b4', '#aec7e8'),  # Pair 1: darker and lighter blue
    ('#ff7f0e', '#ffbb78'),  # Pair 2: darker and lighter orange
    ('#2ca02c', '#98df8a'),  # Pair 3: darker and lighter green
    ('#d62728', '#ff9896'),  # Pair 4: darker and lighter red
    ('#9467bd', '#c5b0d5'),  # Pair 5: darker and lighter purple
    ('#8c564b', '#c49c94'),  # Pair 6: darker and lighter brown
    ('#e377c2', '#f7b6d2'),  # Pair 7: darker and lighter pink
    ('#7f7f7f', '#c7c7c7'),  # Pair 8: darker and lighter gray
    ('#bcbd22', '#dbdb8d'),  # Pair 9: darker and lighter yellow-green
    ('#17becf', '#9edae5')   # Pair 10: darker and lighter teal
]
# distinct_colors = [
#     ('#1f77b4', '#aec7e8'),  # Pair 1: darker and lighter blue
#     ('#ff7f0e', '#ffbb78'),  # Pair 2: darker and lighter orange
#     ('#2ca02c', '#98df8a'),  # Pair 3: darker and lighter green
#     ('#d62728', '#ff9896'),  # Pair 4: darker and lighter red
#     ('#9467bd', '#c5b0d5'),  # Pair 5: darker and lighter purple
# ]

# Assign colors to each experiment pair
color_mapping = {}
pair_index = 0
for experiment in data.index:
    if "Train" in experiment:
        color_mapping[experiment] = distinct_colors[pair_index][1]
    # elif "Eval" in experiment:
        color_mapping[experiment.replace('Train', 'Eval')] = distinct_colors[pair_index][0]
        pair_index = (pair_index + 1) % len(distinct_colors)

# Plot individual histograms of metrics for each experiment
metrics_to_plot = ['overall_acc', 'class_acc_fail', 'class_acc_success', 'class_acc_fnr', 'class_acc_fpr']
for metric in metrics_to_plot:
    plt.figure(figsize=(12, 6))
    bars = []
    for index, value in data[metric].items():
        color = color_mapping[index]
        bars.append(plt.bar(index, value, color=color, edgecolor='black'))
    
    plt.ylabel('Value', fontsize=10)
    plt.title(metric.replace('_', ' ').title(), fontsize=14)
    plt.ylim(0, 1.1)

    # Automatically adjust x-axis labels to avoid overlap
    labels = data.index
    max_label_length = 10
    new_labels = [label if len(label) <= max_label_length else '\n'.join(label.split()) for label in labels]
    plt.xticks(ticks=np.arange(len(labels)), labels=new_labels, rotation=0, fontsize=8, ha='center')
    plt.yticks(fontsize=8)

    # Annotate values on top of the bars
    for bar_group in bars:
        for bar in bar_group:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

# for metric in metrics_to_plot:
#     plt.figure(figsize=(12, 6))
#     bars = []
#     for index, value in data[metric].items():
#         color = color_mapping[index]
#         bars.append(plt.bar(index, value, color=color, edgecolor='black'))
    
#     plt.ylabel('Value', fontsize=10)
#     plt.title(metric.replace('_', ' ').title(), fontsize=14)
#     plt.ylim(0, 1.1)
#     plt.xticks(rotation=45, fontsize=8)
#     plt.yticks(fontsize=8)

#     # Annotate values on top of the bars
#     for bar_group in bars:
#         for bar in bar_group:
#             yval = bar.get_height()
#             plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)

#     plt.tight_layout()
#     plt.show()

# Plot histograms of metrics for each experiment
# metrics_to_plot = ["overall_acc", "class_acc_fail", "class_acc_success", "class_acc_fnr", "class_acc_fpr"]
# # plt.figure(figsize=(15, 20))
# fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(15, 20), constrained_layout=True)
# for ax, metric in zip(axes, metrics_to_plot):
#     ax.bar(data.index, data[metric], color='skyblue', edgecolor='black')
#     ax.set_ylabel('Value', fontsize=12)
#     ax.set_title(metric.replace('_', ' ').title(), fontsize=15)
#     ax.set_ylim(0, 1)
#     ax.tick_params(axis='x', rotation=45, labelsize=10)
#     ax.tick_params(axis='y', labelsize=10)
# plt.show()
# Plot individual histograms of metrics for each experiment
# metrics_to_plot = ['overall_acc', 'class_acc_fail', 'class_acc_success', 'class_acc_fnr', 'class_acc_fpr']

# for metric in metrics_to_plot:
#     plt.figure(figsize=(12, 6))
#     bars = plt.bar(data.index, data[metric], color='skyblue', edgecolor='black')
#     plt.ylabel('Value', fontsize=12)
#     plt.title(metric.replace('_', ' ').title(), fontsize=15)
#     plt.ylim(0, 1)
#     plt.xticks(rotation=45, fontsize=10)
#     plt.yticks(fontsize=10)

#     # Annotate values on top of the bars
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

#     plt.tight_layout()
#     plt.show()