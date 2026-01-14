##FOR ADVERSARIAL PERTURBATION，results saved under ./ag-news and ./sst2
from sklearn.metrics import classification_report
import json

attack_method = "pruthi"
dataset_name = "sst2"
defense = "gector_result"

y_true,y_pred = [],[]
with open(f"../onion_result/{dataset_name}/{attack_method}.json", "r", encoding="utf-8") as f:
    results = json.load(f)

with open(f"./{dataset_name}/{attack_method}.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()

for idx,line in enumerate(lines):
    gec_output = line.strip().lower().split()
    original_output = results[idx]["perturbed_text"].replace("[","").replace("]","").lower().split()
    y_pred += [0 if x == y else 1 for x,y in zip(gec_output,original_output)]           ###if the word is changed means localised successfully
    y_true += results[idx]["target"]

    if len(results[idx]["target"]) != len([0 if x == y else 1 for x,y in zip(gec_output,original_output)]): ###if there exists exception, then print the line
        print(line)

report = classification_report(y_true, y_pred, output_dict=True)

with open(f"./{dataset_name}/{attack_method}_f1pr_sklearn.json", "w+", encoding="utf-8") as fp:
    fp.write(json.dumps(report))



# ##FOR BACKDOOR TRIGGERS，results saved under ./on_badnet
# from sklearn.metrics import classification_report
# import json
#
# dataset_name = "offenseval"         ###offenseval，sst2
# defense = "gector_result"
#
# y_true,y_pred = [],[]
# with open(f"../onion_result/on_badnet/{dataset_name}/train.json", "r", encoding="utf-8") as f:
#     results = json.load(f)
#
# with open(f"./on_badnet/{dataset_name}/train.txt", 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#
#
# for idx,line in enumerate(lines):
#     gec_output = line.strip().lower().split()
#     original_output = results[idx]["perturbed_text"].replace("[","").replace("]","").lower().split()
#     y_pred += [0 if x == y else 1 for x,y in zip(gec_output,original_output)]           ###if the word is changed means localised successfully
#     y_true += results[idx]["target"]
#
#     if len(results[idx]["target"]) != len([0 if x == y else 1 for x,y in zip(gec_output,original_output)]):     ###if there exists exception, then print the line
#         print(results[idx]["perturbed_text"])
#
# report = classification_report(y_true, y_pred, output_dict=True)
#
# with open(f"./on_badnet/{dataset_name}/train_f1pr_sklearn.json", "w+", encoding="utf-8") as fp:
#     fp.write(json.dumps(report))




# ##FOR GRAMMATICAL ERRORS，results saved under ./on_GEC_FCE
# defense = "gector_result"
#
# y_true,y_pred = [],[]
# with open(f"../onion_result/on_GEC_FCE/test.json", "r", encoding="utf-8") as f:
#     results = json.load(f)
#
# with open(f"./on_GEC_FCE/test.txt", 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#
# for idx,line in enumerate(lines):
#     gec_output = line.strip().lower().split()
#     original_output = results[idx]["perturbed_text"].replace("[","").replace("]","").lower().split()
#     y_pred += [0 if x == y else 1 for x,y in zip(gec_output,original_output)]           ###if the word is changed means localised successfully
#     y_true += results[idx]["target"]
#
#     if len(results[idx]["target"]) != len([0 if x == y else 1 for x,y in zip(gec_output,original_output)]):       ###if there exists exception, then print the line
#         print(line)
#
# report = classification_report(y_true, y_pred, output_dict=True)
#
# with open(f"./on_GEC_FCE/test_f1pr.json", "w+", encoding="utf-8") as fp:
#     fp.write(json.dumps(report))



# ##FOR ADVERSARIAL PROMPTS，results saved under ./on_prompt
# dataset_name = "on_prompt"
# defense = "gector_result"
#
# for attack_method in ["bertattack", "checklist", "deepwordbug", "stresstest", "textfooler", "textbugger"]:
#     y_true,y_pred = [],[]
#     with open(f"../onion_result/{dataset_name}/{attack_method}.json", "r", encoding="utf-8") as f:
#         results = json.load(f)
#
#     with open(f"./{dataset_name}/{attack_method}.txt", 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#
#     for idx,line in enumerate(lines):
#         gec_output = line.strip().lower().split()
#         original_output = results[idx]["perturbed_text"].lower().split()
#         y_pred += [0 if x == y else 1 for x,y in zip(gec_output,original_output)]           ###if the word is changed means localised successfully
#         y_true += results[idx]["target"]
#
#         if len(results[idx]["target"]) != len([0 if x == y else 1 for x,y in zip(gec_output,original_output)]):     ###if there exists exception, then print the line
#             print(line)
#
#     report = classification_report(y_true, y_pred, output_dict=True)
#
#     with open(f"./{dataset_name}/{attack_method}_f1pr_sklearn.json", "w+", encoding="utf-8") as fp:
#         fp.write(json.dumps(report))