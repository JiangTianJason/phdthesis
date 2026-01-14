###Section 5.3.3, Figure 5.8-5.11###

####combination_result，FOR ADVERSARIAL，results saved under ./ag-news and ./sst2
from sklearn.metrics import classification_report
import json,os

attack_method = "bae"
dataset_name = "ag-news"

for defense_1 in ["fgws_result","xgboost_result","our_result","bfclass_result","onion_result","rank_result","gector_standard_output_result"]:
    result_json_path_1 = rf"../{defense_1}/{dataset_name}/{attack_method}.json"
    if not os.path.isfile(result_json_path_1):
        if defense_1 != "our_result":
            result_json_path_1 = rf"../{defense_1}/{attack_method}_{dataset_name}_bert.json"
        else:
            result_json_path_1 = rf"../{defense_1}/only_perplexity_0_probability_0.01/{dataset_name}/{attack_method}.json"

    for defense_2 in ["fgws_result","xgboost_result","our_result","bfclass_result","onion_result","rank_result","gector_standard_output_result"]:
        if defense_1 != defense_2:
            result_json_path_2 = rf"../{defense_2}/{dataset_name}/{attack_method}.json"
            if not os.path.isfile(result_json_path_2):
                if defense_2 != "our_result":
                    result_json_path_2 = rf"../{defense_2}/{attack_method}_{dataset_name}_bert.json"
                else:
                    result_json_path_2 = rf"../{defense_2}/only_perplexity_0_probability_0.01/{dataset_name}/{attack_method}.json"


            with open(result_json_path_1, "r", encoding="utf-8") as f:

                results = json.load(f)

            y_true,y_pred = [],[]
            for item in results:
                y_pred += item["pred"]
                y_true += item["target"]

            with open(result_json_path_2, "r", encoding="utf-8") as f:

                results = json.load(f)

            y_true_onion,y_pred_onion = [],[]
            for item in results:
                y_pred_onion += item["pred"]
                y_true_onion += item["target"]

            y_pred_and =  [a & b for a, b in zip(y_pred, y_pred_onion)]
            result_and = classification_report(y_true, y_pred_and, output_dict=True)

            y_pred_or =  [a | b for a, b in zip(y_pred, y_pred_onion)]
            result_or = classification_report(y_true, y_pred_or, output_dict=True)

            save_path = fr"./{dataset_name}/{attack_method}_f1pr_sklearn.json"
            with open(save_path, "a+",
                      encoding="utf-8") as fp:
                fp.write(f"{defense_1} & {defense_2}: "+ json.dumps(result_and) + '\n')
                fp.write(f"{defense_1} | {defense_2}: " + json.dumps(result_or) + '\n')
                fp.write("\n")

print(f"Saved Successfully!")



# ####combination_result，FOR BACKDOOR TRIGGERS，results saved under ./on_badnet
# from sklearn.metrics import classification_report
# import json
#
# dataset_name = "sst2"
#
# for defense_1 in ["fgws_result","xgboost_result","our_result","bfclass_result","onion_result","rank_result","gector_standard_output_result"]:
#     if defense_1 != "our_result":
#         result_json_path_1 = rf"../{defense_1}/on_badnet/{dataset_name}/test.json"
#     else:
#         result_json_path_1 = rf"../{defense_1}/only_perplexity_0_probability_0.01/on_badnet/{dataset_name}/test.json"
#
#     for defense_2 in ["fgws_result","xgboost_result","our_result","bfclass_result","onion_result","rank_result","gector_standard_output_result"]:
#         if defense_1 != defense_2:
#             if defense_2 != "our_result":
#                 result_json_path_2 = rf"../{defense_2}/on_badnet/{dataset_name}/test.json"
#             else:
#                 result_json_path_2 = rf"../{defense_2}/only_perplexity_0_probability_0.01/on_badnet/{dataset_name}/test.json"
#
#
#             with open(result_json_path_1, "r", encoding="utf-8") as f:
#
#                 results = json.load(f)
#
#             y_true,y_pred = [],[]
#             for item in results:
#                 y_pred += item["pred"]
#                 y_true += item["target"]
#
#             with open(result_json_path_2, "r", encoding="utf-8") as f:
#
#                 results = json.load(f)
#
#             y_true_onion,y_pred_onion = [],[]
#             for item in results:
#                 y_pred_onion += item["pred"]
#                 y_true_onion += item["target"]
#
#             y_pred_and =  [a & b for a, b in zip(y_pred, y_pred_onion)]
#             result_and = classification_report(y_true, y_pred_and, output_dict=True)
#
#             y_pred_or =  [a | b for a, b in zip(y_pred, y_pred_onion)]
#             result_or = classification_report(y_true, y_pred_or, output_dict=True)
#
#             save_path = fr"./on_badnet/{dataset_name}/test_new_f1pr_sklearn.json"
#             with open(save_path, "a+",
#                       encoding="utf-8") as fp:
#                 fp.write(f"{defense_1} & {defense_2}: "+ json.dumps(result_and) + '\n')
#                 fp.write(f"{defense_1} | {defense_2}: " + json.dumps(result_or) + '\n')
#                 fp.write("\n")
#
# print(f"Saved Successfully!")



# ####combination_result，FOR GRAMMATICAL ERRORS，results saved under ./on_GEC_FCE
# from sklearn.metrics import classification_report
# import json
#
# for defense_1 in ["fgws_result","our_result","bfclass_result","onion_result","rank_result","gector_standard_output_result"]:
#     if defense_1 != "our_result":
#         result_json_path_1 = rf"../{defense_1}/on_GEC_FCE/test.json"
#     else:
#         result_json_path_1 = rf"../{defense_1}/only_perplexity_0_probability_0.01/on_GEC_FCE/test.json"
#
#     for defense_2 in ["fgws_result","our_result","bfclass_result","onion_result","rank_result","gector_standard_output_result"]:
#         if defense_1 != defense_2:
#             if defense_2 != "our_result":
#                 result_json_path_2 = rf"../{defense_2}/on_GEC_FCE/test.json"
#             else:
#                 result_json_path_2 = rf"../{defense_2}/only_perplexity_0_probability_0.01/on_GEC_FCE/test.json"
#
#
#             with open(result_json_path_1, "r", encoding="utf-8") as f:
#
#                 results = json.load(f)
#
#             y_true,y_pred = [],[]
#             for item in results:
#                 y_pred += item["pred"]
#                 y_true += item["target"]
#
#             with open(result_json_path_2, "r", encoding="utf-8") as f:
#
#                 results = json.load(f)
#
#             y_true_onion,y_pred_onion = [],[]
#             for item in results:
#                 y_pred_onion += item["pred"]
#                 y_true_onion += item["target"]
#
#             y_pred_and =  [a & b for a, b in zip(y_pred, y_pred_onion)]
#             result_and = classification_report(y_true, y_pred_and, output_dict=True)
#
#             y_pred_or =  [a | b for a, b in zip(y_pred, y_pred_onion)]
#             result_or = classification_report(y_true, y_pred_or, output_dict=True)
#
#             save_path = fr"./on_GEC_FCE/test_f1pr_sklearn.json"
#             with open(save_path, "a+",
#                       encoding="utf-8") as fp:
#                 fp.write(f"{defense_1} & {defense_2}: "+ json.dumps(result_and) + '\n')
#                 fp.write(f"{defense_1} | {defense_2}: " + json.dumps(result_or) + '\n')
#                 fp.write("\n")
#
# print(f"Saved Successfully!")



# ####combination_result，FOR PROMPT ADVERSARIAL，results saved under ./on_prompt
# from sklearn.metrics import classification_report
# import json
#
# attack_method = "textbugger"
#
# for defense_1 in ["our_result","bfclass_result","onion_result","rank_result","gector_standard_output_result"]:
#     if defense_1 != "our_result":
#         result_json_path_1 = rf"../{defense_1}/on_prompt/{attack_method}.json"
#     else:
#         result_json_path_1 = rf"../{defense_1}/only_perplexity_0_probability_0.01/on_prompt/{attack_method}.json"
#
#     for defense_2 in ["our_result","bfclass_result","onion_result","rank_result","gector_standard_output_result"]:
#         if defense_1 != defense_2:
#             if defense_2 != "our_result":
#                 result_json_path_2 = rf"../{defense_2}/on_prompt/{attack_method}.json"
#             else:
#                 result_json_path_2 = rf"../{defense_2}/only_perplexity_0_probability_0.01/on_prompt/{attack_method}.json"
#
#
#             with open(result_json_path_1, "r", encoding="utf-8") as f:
#
#                 results = json.load(f)
#
#             y_true,y_pred = [],[]
#             for item in results:
#                 y_pred += item["pred"]
#                 y_true += item["target"]
#
#             with open(result_json_path_2, "r", encoding="utf-8") as f:
#
#                 results = json.load(f)
#
#             y_true_onion,y_pred_onion = [],[]
#             for item in results:
#                 y_pred_onion += item["pred"]
#                 y_true_onion += item["target"]
#
#             y_pred_and =  [a & b for a, b in zip(y_pred, y_pred_onion)]
#             result_and = classification_report(y_true, y_pred_and, output_dict=True)
#
#             y_pred_or =  [a | b for a, b in zip(y_pred, y_pred_onion)]
#             result_or = classification_report(y_true, y_pred_or, output_dict=True)
#
#             save_path = fr"./on_prompt/{attack_method}_f1pr_sklearn.json"
#             with open(save_path, "a+",
#                       encoding="utf-8") as fp:
#                 fp.write(f"{defense_1} & {defense_2}: "+ json.dumps(result_and) + '\n')
#                 fp.write(f"{defense_1} | {defense_2}: " + json.dumps(result_or) + '\n')
#                 fp.write("\n")
#
# print(f"Saved Successfully!")