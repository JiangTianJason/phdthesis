import matplotlib.pyplot as plt
import json,os
import seaborn as sns
import pandas as pd
import numpy as np

def calculate_draw_Interval(ax,title_name,perp_prob,target_type):
    patches = ax.patches

    density_values = [patch.get_height() for patch in patches]
    bin_edges = [patch.get_x() for patch in patches]
    bin_widths = [patch.get_width() for patch in patches]

    max_density_idx = np.argmax(density_values)
    max_density_value = density_values[max_density_idx]

    max_density_interval = (
        bin_edges[max_density_idx],
        bin_edges[max_density_idx] + bin_widths[max_density_idx]
    )

    print(title_name + " " + target_type +" " + perp_prob)
    print(f"最大密度区间: [{max_density_interval[0]:}, {max_density_interval[1]:})")

    with open(r"./max_density_interval.txt","a+") as f:
        f.write(title_name + " " + target_type +" " + perp_prob + "\n")
        f.write(f"最大密度区间: [{max_density_interval[0]:}, {max_density_interval[1]:})" + "\n")
        f.write("\n")


def load_json_data(path):
    label_all, pred_all,x_perturbation,y_perturbation,x_benign,y_benign = [], [], [], [], [], []
    for single_path in path:
        for root, dirs, files in os.walk(single_path):
            for file in files:
                if file.endswith(".json"):
                    print(os.path.join(root, file))
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        results = json.load(f)

                    for item in results:
                        pred_all += item["pred"]
                        label_all += item["target"]
                    for pred, label in zip(pred_all, label_all):
                        if label == 0 and not np.isnan(pred[0]):
                            x_benign.append(pred[0])
                            y_benign.append(pred[1])
                        elif label == 1 and not np.isnan(pred[0]):
                            x_perturbation.append(pred[0])  ###Perplexity
                            y_perturbation.append(pred[1])  ###Probability

    return x_benign,y_benign,x_perturbation,y_perturbation


if __name__ == "__main__":

    backdoor_file_path = [r"./on_badnet/offenseval",r"./on_badnet/sst2"]
    adversarial_file_path = [r"./sst2",r"./ag-news"]
    prompt_file_path = [r"./on_prompt"]
    grammar_file_path = [r"on_GEC_FCE"]

    file_path = adversarial_file_path

    if file_path == adversarial_file_path:
        title_name = "Adversarial"
    elif file_path == backdoor_file_path:
        title_name = "Backdoor"
    elif file_path == grammar_file_path:
        title_name = "Grammar"
    else:
        title_name = "Prompt"

    perp_benign, prob_benign, perp_perturbation, prob_perturbation = load_json_data(file_path)

    ### 1
    sns.histplot(perp_benign, label="normal", color="blue", kde=False, stat="density")
    sns.histplot(perp_perturbation, label="anomaly", color="orange", kde=False, stat="density")
    plt.xlim([-200,200])
    plt.xlabel("Perplexity",fontsize=15)
    plt.ylabel("Density",fontsize=15)

    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.legend(fontsize=15)
    plt.grid(alpha = 0.3,linestyle='--')
    plt.savefig(rf'./{title_name}_Perplexity.jpg', dpi=300)
    plt.show()

    # # ### 2
    # sns.histplot(prob_benign,label="normal",color = "blue",kde=False,stat="density")
    # sns.histplot(prob_perturbation,label="anomaly",color="orange",kde=False,stat="density",bins=20)
    # plt.xlim([0,1])
    # plt.xlabel("Probability",fontsize=15)
    # plt.ylabel("Density",fontsize=15)

    # plt.tick_params(axis='x', labelsize=14)
    # plt.tick_params(axis='y', labelsize=14)
    #
    # plt.tight_layout()
    # plt.legend(fontsize=15)
    # plt.grid(alpha=0.3, linestyle='--')
    # plt.savefig(rf'./{title_name}_Probability.jpg', dpi=300)
    # plt.show()


    # ### 3
    # ax_normal = sns.histplot(perp_benign, label="normal", color="blue", kde=False, stat="density")
    # calculate_draw_Interval(ax_normal,title_name,"perplexity","normal")

    # ### 4
    # ax_anomaly = sns.histplot(perp_perturbation, label="anomaly", color="orange", kde=False, stat="density")
    # calculate_draw_Interval(ax_anomaly,title_name,"perplexity","anomaly")

    # ### 5
    # ax_normal_prob = sns.histplot(prob_benign,label="normal",color = "blue",kde=False,stat="density")
    # calculate_draw_Interval(ax_normal_prob,title_name,"probability","normal")

    # ### 6
    # ax_anomaly_prob = sns.histplot(prob_perturbation,label="anomaly",color="orange",kde=False,stat="density",bins=20)
    # calculate_draw_Interval(ax_anomaly_prob,title_name,"probability","anomaly")