import matplotlib.pyplot as plt
import numpy as np

metrics = ['Precision', 'Recall', 'F1 score']
datasets = ['agnews', 'sst-2']
methods = ['bae', 'pruthi', 'textfooler']
colors = {'Precision': '#1f77b4', 'Recall': '#ff7f0e', 'F1 score': '#2ca02c'}
bar_width = 0.25
x_offset = {'Precision': -bar_width, 'Recall': 0, 'F1 score': bar_width}

data = {
    'agnews': {
        'bae': {'Precision': (24.06,25.75), 'Recall': (52.49,46.00), 'F1 score': (33.00,33.02)},
        'pruthi':  {'Precision': (18.97,21.80), 'Recall': (93.69,93.38), 'F1 score': (31.55,35.34)},
        'textfooler': {'Precision': (58.88,62.31), 'Recall': (60.78,60.22), 'F1 score': (59.82,61.25)}
    },
    'sst-2': {
        'bae': {'Precision': (16.68,18.16), 'Recall': (34.06,30.05), 'F1 score': (22.39,22.64)},
        'pruthi':  {'Precision': (25.90,30.24), 'Recall': (87.92,87.78), 'F1 score': (40.01,44.98)},
        'textfooler': {'Precision': (36.47,41.22), 'Recall': (60.70,60.48), 'F1 score': (45.56,49.03)}
    }
}

fig, axes = plt.subplots(1, 2, figsize=(18, 5))

for ax, dataset in zip(axes, datasets):
    x_base = np.arange(len(methods))

    for i, method in enumerate(methods):
        for metric in metrics:
            x_pos = x_base[i] + x_offset[metric]
            original = data[dataset][method][metric][0]
            ax.bar(x_pos, original, width=bar_width - 0.05,
                   color=colors[metric], alpha=0.6, edgecolor='black')

            ax.text(x_pos, original + 0.01,
                    f'{original:.2f}',
                    ha='center', va='bottom',
                    fontsize=11, color='black')

    for i, method in enumerate(methods):
        x_points = [x_base[i] + x_offset[metrics[0]],x_base[i] + x_offset[metrics[1]],x_base[i] + x_offset[metrics[2]]]
        y_points = [
            data[dataset][method][metrics[0]][1],
            data[dataset][method][metrics[1]][1],
            data[dataset][method][metrics[2]][1]
        ]
        ax.plot(x_points, y_points, marker='o', markersize=5,
                color="black", linestyle='--', linewidth=1)

    ax.set_xticks(x_base)
    ax.set_xticklabels(methods)
    ax.set_xlabel("Attack Method",fontsize=15)
    ax.set_ylabel("Values",fontsize=15)
    ax.set_title(f'{dataset}', fontsize=15, pad=10,fontweight='bold')

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # ax.set_ylim(0, 1.0)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

handles = [
    plt.Rectangle((0, 0), 1, 1, fc=colors['Precision'], alpha=0.6, edgecolor='black'),
    plt.Rectangle((0, 0), 1, 1, fc=colors['Recall'], alpha=0.6, edgecolor='black'),
    plt.Rectangle((0, 0), 1, 1, fc=colors['F1 score'], alpha=0.6, edgecolor='black'),
    plt.Line2D([], [], color='black', marker='o', linestyle='--', markersize=5,linewidth=1)
]
labels = ['Precision($FT-bert$)',"Recall($FT-bert$)","F1 score($FT-bert$)","($bert$)"]
fig.legend(handles, labels, loc='lower center',ncol=4, fontsize=15)

plt.tight_layout(rect=[0, 0.1, 1, 1])
# # plt.subplots_adjust(top=0.8)
plt.savefig('./适配性计算模型变化后的对抗扰动定位效果.jpg', dpi=300)
plt.show()