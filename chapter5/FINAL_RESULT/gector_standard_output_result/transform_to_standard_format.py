###"This is one of Polanski 's only films ." -----> {"pred": [0, 0, 0, 0, 0, 0, 0, 0], "target": [0, 0, 0, 0, 0, 0, 1, 0], "perturbed_text": "this is one of polanski 's [[only]] films"}

##FOR ADVERSARIAL PERTURBATION，results saved under ./ag-news and ./sst2
import json

attack_method = "textfooler"
dataset_name = "ag-news"
defense = "gector_result"

with open(fr"../onion_result/{dataset_name}/{attack_method}.json", "r", encoding="utf-8") as f:
    results = json.load(f)

with open(fr"../{defense}/{dataset_name}/{attack_method}.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()

final_result = []
for idx,line in enumerate(lines):
    gec_output = line.strip().lower().split()
    original_output = results[idx]["perturbed_text"].replace("[","").replace("]","").lower().split()

    if len(results[idx]["target"]) != len([0 if x == y else 1 for x,y in zip(gec_output,original_output)]):
        print(line)

    final_result.append({"pred": [0 if x == y else 1 for x,y in zip(gec_output,original_output)], "target": results[idx]["target"], "perturbed_text": results[idx]["perturbed_text"]})

with open(fr'./{dataset_name}/{attack_method}.json', "w+", encoding="utf-8") as f:
    f.write(json.dumps(final_result))



# ##FOR BACKDOOR TRIGGERS，results saved under ./on_badnet
# import json
#
# dataset_name = "sst2"         ###offenseval，sst2
# defense = "gector_result"
#
# y_true,y_pred = [],[]
# with open(fr"../onion_result/on_badnet/{dataset_name}/test.json", "r", encoding="utf-8") as f:
#     results = json.load(f)
#
# with open(fr"../{defense}/on_badnet/{dataset_name}/test.txt", 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#
# final_result = []
# for idx,line in enumerate(lines):
#     gec_output = line.strip().lower().split()
#     original_output = results[idx]["perturbed_text"].replace("[","").replace("]","").lower().split()
#     final_result.append(
#         {"pred": [0 if x == y else 1 for x, y in zip(gec_output, original_output)], "target": results[idx]["target"],
#          "perturbed_text": results[idx]["perturbed_text"]})
#
# with open(
#         fr'./on_badnet/{dataset_name}/test.json',
#         "w+", encoding="utf-8") as f:
#     f.write(json.dumps(final_result))



# ##FOR GRAMMATICAL ERRORS，results saved under ./on_GEC_FCE
# import json
#
# defense = "gector_result"
#
# y_true,y_pred = [],[]
# with open(fr"../onion_result/on_GEC_FCE/test.json", "r", encoding="utf-8") as f:
#     results = json.load(f)
#
# with open(fr"../{defense}/on_GEC_FCE/test.txt", 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#
# final_result = []
# for idx,line in enumerate(lines):
#     gec_output = line.strip().lower().split()
#     original_output = results[idx]["perturbed_text"].replace("[","").replace("]","").lower().split()
#     final_result.append(
#         {"pred": [0 if x == y else 1 for x, y in zip(gec_output, original_output)], "target": results[idx]["target"],
#          "perturbed_text": results[idx]["perturbed_text"]})
#
# with open(
#         fr'./on_GEC_FCE/test.json',
#         "w+", encoding="utf-8") as f:
#     f.write(json.dumps(final_result))



# ##FOR ADVERSARIAL PROMPTS，results saved under ./on_prompt
# import json
#
# dataset_name = "on_prompt"
# defense = "gector_result"
#
# for attack_method in ["bertattack", "checklist", "deepwordbug", "stresstest", "textfooler", "textbugger"]:
#     y_true,y_pred = [],[]
#     with open(fr"../onion_result/{dataset_name}/{attack_method}.json", "r", encoding="utf-8") as f:
#         results = json.load(f)
#
#     with open(fr"../{defense}/{dataset_name}/{attack_method}.txt", 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#
#     final_result = []
#     for idx,line in enumerate(lines):
#         gec_output = line.strip().lower().split()
#         original_output = results[idx]["perturbed_text"].lower().split()
#         final_result.append(
#             {"pred": [0 if x == y else 1 for x, y in zip(gec_output, original_output)], "target": results[idx]["target"],
#              "perturbed_text": results[idx]["perturbed_text"]})
#
#     with open(
#             fr'./on_prompt/{attack_method}.json',
#             "w+", encoding="utf-8") as f:
#         f.write(json.dumps(final_result))