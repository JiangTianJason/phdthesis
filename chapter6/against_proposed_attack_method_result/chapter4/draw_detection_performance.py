import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

categories = ['Morpheus', 'DeepWordBug', 'TextFooler\n(Word2Vec)', 'TextFooler','BAE\n(CyBERT)', 'BAE', 'Ours']
values1 = [10.00,89.42,69.23,76.70,68.81,62.00,70.37]    ###BERT
values2 = [41.67,83.52,67.80,73.47,70.80,60.42,69.89]    ###ALBERT
values3 = [20.00,82.98,47.37,67.86,77.71,67.57,52.63]    ###XLM-RoBERTa

print((sum(values1) + sum(values2) + sum(values3)) / 21)

plt.figure(figsize=(16, 6))
# plt.tight_layout()

bar_width = 0.3
index = np.arange(len(categories))

plt.bar(index, values1, bar_width, label='BERT',color = "#5E8BC5")

plt.bar(index + bar_width, values2, bar_width, label='ALBERT', color = "#D4695E")

plt.bar(index + 2 * bar_width, values3, bar_width, label='XLM-RoBERTa', color = "#DCA11D")

plt.xlabel('Methods')
plt.ylabel('Accuracy (%)')

plt.legend(loc='upper left')

plt.xticks(index + bar_width, categories,fontproperties = FontProperties(weight='bold'))
plt.ylim(bottom=0)
plt.grid(True, linestyle='dashed',axis = "y", color='gray', alpha=0.5)

x = np.arange(len(categories))
for i in range(len(categories)):
    plt.text(x[i], values1[i] + 0.5, str(values1[i]), ha='center', va='bottom')
    plt.text(x[i] + bar_width, values2[i] + 0.5, str(values2[i]), ha='center', va='bottom')
    plt.text(x[i] + 2 * bar_width, values3[i] + 0.5, str(values3[i]), ha='center', va='bottom')

plt.savefig(fr'./detection_by_stylistic_representatino.jpg', dpi=300,bbox_inches='tight')
plt.show()