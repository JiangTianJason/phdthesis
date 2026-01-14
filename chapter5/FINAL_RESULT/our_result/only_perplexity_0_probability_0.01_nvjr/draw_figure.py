import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import random


x_label = ['bae \n (agnews)','pruthi \n (agnews)','textfooler \n (agnews)','bae \n (sst-2)','pruthi \n (sst-2)','textfooler \n (sst-2)', 'bertattack \n (PromptBench)','checklist \n (PromptBench)','deepwordbug \n (PromptBench)','stresstest \n (PromptBench)','textbugger \n (PromptBench)','textfooler \n (PromptBench)']
nvjr_words = [26.31,29.94,54.31,21.02,44.23,43.19,50.68,53.09,73.19,6.23,68.16,67.71]
all_words = [33.02,35.34,61.25,22.64,44.98,49.03,57.14,59.26,74.01,10.77,68.48,66.44]

x = np.arange(len(x_label))
width = 0.3

plt.figure(figsize=(20,8),dpi=80)

rects1=plt.bar(x-width/2,nvjr_words,width=width,label='POSs')

rects2=plt.bar(x+width/2,all_words,width=width,label='All')

plt.legend(loc='upper left', fontsize=15)
plt.grid(alpha=0.3)
plt.xlabel('Attack scenario',fontsize=15)
plt.ylabel('F1 score',fontsize=15)
plt.xticks(x,x_label)


plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=14)

for label in plt.gca().get_xticklabels():
    label.set_fontweight('bold')

for rect1 in rects1:
    height = rect1.get_height()
    plt.text(rect1.get_x()+rect1.get_width()/2,height+1,str(height),ha='center',fontsize=11)

for rect2 in rects2:
    height = rect2.get_height()
    plt.text(rect2.get_x()+rect2.get_width()/2,height+1,str(height),ha='center',fontsize=11)

plt.tight_layout()
plt.savefig('词性引导与全量计算方式的定位性能比较.jpg', dpi=300)
plt.show()



nvjr_words = [26.31,29.94,54.31,21.02,44.23,43.19,50.68,53.09,73.19,6.23,68.16,67.71]
bfclass_words = [25.19,2.25,21.56,16.47,2.23,51.33,28.72,12.69,0,16.92,1.81,25]
onion_words = [22.39,39.96,45.55,33,31.55,59.82,55.86,53.39,72.10,12.86,66.28,64.81]
rank_words = [18.83,28.45,40.54,25.21,23.21,50.80,43.73,31.86,50.04,15.98,52.44,56.45]
gector_words = [12.29,25.06,25.74,15.06,9.05,30.79,20.30,3.28,54.40,5.89,34.44,28.10]

if len(nvjr_words) == len(gector_words):
    print((np.sum(nvjr_words) - np.sum(gector_words)) / len(nvjr_words))
    # subtract = np.subtract(all_words,nvjr_words)
    # print(np.mean(subtract))
