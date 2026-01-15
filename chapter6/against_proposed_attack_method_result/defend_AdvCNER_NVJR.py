###Table 5.9
###Test the anomaly detection performance on "NV" in Chapter 3 and on 7 attack methods in Chapter 4

import os
import json
import math
from sklearn.svm import SVC
from typing import List, Union
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances, roc_auc_score, confusion_matrix, classification_report
from transformers import AutoModel, AutoTokenizer,AutoConfig,pipeline
from tqdm import tqdm

np.random.seed(42)

device = torch.device("cuda")
model = AutoModel.from_pretrained(r"/root/autodl-tmp/LUAR",trust_remote_code=True)      ###The model is saved locally
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(r"/root/autodl-tmp/LUAR",add_special_tokens=False)

def get_uar_embedding(sample: Union[List[str], str]):
      if isinstance(sample, str):
        sample = [sample]

      tok = tokenizer.batch_encode_plus(
          sample,
          truncation=False,
          padding=True,
          return_tensors="pt"
        )

      # UAR's backbone can handle up to 512 tokens
      # Here we're padding the sample to the nearest multiple of 512:
      _, NT = tok["input_ids"].size()
      nearest = 512 * int(math.ceil(NT / 512))
      tok["input_ids"] = F.pad(tok["input_ids"], (1, nearest - NT - 1), value=tokenizer.pad_token_id)
      tok["attention_mask"] = F.pad(tok["attention_mask"], (1, nearest - NT - 1), value=0)

      # Reshape into (batch_size=1, history_size=N, num_tokens=512)
      tok["input_ids"] = tok["input_ids"].reshape(1, -1, 512).to(device)
      tok["attention_mask"] = tok["attention_mask"].reshape(1, -1, 512).to(device)

      with torch.inference_mode():
        out = model(**tok)
        out = F.normalize(out, p=2.0)
      return out



# ########################################Anomaly detection for all 7 attack methods in Chapter 4########################################################
# for victim_model in ["bert","albert","xlm"]:
#     for method in ["bae","bae_cybert","deepwordbug","morpheus","textfooler","textfooler_word2vec","nvjr"]:
#
#         ###Load TRAIN data contained Adversarial examples with their corresponding original samples
#         filename = rf"./chapter4/adversarial_training/output_{victim_model}/cti-{method}-strict-preserve.json"
#         success_count,original_embedding_list,perturbed_embedding_list = 0,[],[]
#         with open(filename, 'r', encoding='utf-8') as fp:
#             json_data = json.load(fp)
#             all_indices = json_data["attacked_examples"]
#             for i in tqdm(all_indices):
#                 if i["status"] == "Successful" and success_count < 100:
#                     success_count += 1
#                     original_embedding_list.append(get_uar_embedding(i["original_text"])[0].cpu().numpy())
#                     perturbed_embedding_list.append(get_uar_embedding(i["perturbed_text"])[0].cpu().numpy())
#
#         embedding_list = original_embedding_list + perturbed_embedding_list
#
#         np.save(fr'./chapter4/{method}/{victim_model}_train_embedding_batch_200.npy', embedding_list)
#         print("Done ->",fr'./chapter4/{method}/{victim_model}_train_embedding_batch_200.npy')
#
#         train_embedding = embedding_list
#         train_labels = [0] * len(original_embedding_list) + [1] * len(perturbed_embedding_list)
#         outlier_detection_model = SVC(random_state=42)
#         outlier_detection_model.fit(train_embedding,train_labels)
#         print("Trained!!!")
#
#         ###Load TEST data contained Adversarial examples with their corresponding original samples
#         filename_test = rf"./chapter4/output_{victim_model}/cti-{method}-strict-preserve.json"
#         original_embedding_list_test,perturbed_embedding_list_test = [],[]
#         with open(filename_test, 'r', encoding='utf-8') as fp:
#             json_data = json.load(fp)
#             all_indices = json_data["attacked_examples"]
#             for i in tqdm(all_indices):
#                 if i["status"] == "Successful":
#                     original_embedding_list_test.append(get_uar_embedding(i["original_text"])[0].cpu().numpy())
#                     perturbed_embedding_list_test.append(get_uar_embedding(i["perturbed_text"])[0].cpu().numpy())
#
#         embedding_list_test = original_embedding_list_test + perturbed_embedding_list_test
#
#         np.save(fr'./chapter4/{method}/{victim_model}_test_embedding_batch.npy', embedding_list_test)
#         print("Done ->",fr'./chapter4/{method}/{victim_model}_test_embedding_batch.npy')
#
#         correct = 0
#         for i,j in zip(original_embedding_list_test,perturbed_embedding_list_test):
#             if outlier_detection_model.predict([i]) == 0 and outlier_detection_model.predict([j]) == 1:
#                 correct += 1
#
#         print("Detection Accuracy: ",correct / len(original_embedding_list_test))
#
#         with open(fr'./chapter4/{method}/{victim_model}.txt', 'a') as f:
#             print("Correct percentage (Detection Accuracy) on adversarial test set and their corresponding original samples: " + str(correct / len(original_embedding_list_test)) + "\n", file=f)



###########################################Anomaly detection for proposed "NV" in Chapter 3#################################################
# for dataset in ["wiki80","tacred"]:
#     for victim_model in ["bert",'bertentity',"pcnn"]:
#
#         ###Load TRAIN data contained Adversarial examples with their corresponding original samples
#         filename_train = fr"./chapter3/adversarial_experiment/{dataset}/{victim_model}/train_NV_{dataset}_{victim_model}.txt"
#         target_sample_in_generate = [eval(line) for line in
#                                      open(filename_train, 'r',
#                                           encoding='utf-8').readlines()]
#
#         train_filepath = rf"./chapter3/dataset/{dataset}/train.txt"
#         original_sample_list = [eval(line) for line in open(train_filepath, 'r',encoding='utf-8').readlines()]
#
#         success_count,original_embedding_list,perturbed_embedding_list = 0,[],[]
#
#         for i in tqdm(target_sample_in_generate):
#             if success_count < 5000:
#                 success_count += 1
#                 original_embedding_list.append(get_uar_embedding(" ".join(original_sample_list[int(i["index"])]["token"]))[0].cpu().numpy())
#                 perturbed_embedding_list.append(get_uar_embedding(" ".join(i["adversary_samples"]["token"]))[0].cpu().numpy())
#
#         embedding_list = original_embedding_list + perturbed_embedding_list
#
#         np.save(fr'./chapter3/nv_{dataset}_{victim_model}_train_embedding_batch_10000.npy', embedding_list)
#         print("Done ->",fr'./chapter3/nv_{dataset}_{victim_model}_train_embedding_batch_10000.npy')
#
#         train_embedding = embedding_list
#         train_labels = [0] * len(original_embedding_list) + [1] * len(perturbed_embedding_list)
#         outlier_detection_model = SVC(random_state=42)
#         outlier_detection_model.fit(train_embedding,train_labels)
#
#
#         ###Load TEST data contained Adversarial examples with their corresponding original samples
#         filename_test = fr"./chapter3/dataset/{dataset}/{victim_model}/noun+verb_{dataset}_{victim_model}_xiawei.txt"
#         target_sample_in_generate = [eval(line) for line in
#                                      open(filename_test, 'r',
#                                           encoding='utf-8').readlines()]
#
#         perturbed_embedding_list_test = []
#
#         for i in tqdm(target_sample_in_generate):
#             perturbed_embedding_list_test.append(get_uar_embedding(" ".join(i["adversary_samples"]["token"]))[0].cpu().numpy())
#
#         np.save(fr'./chapter3/nv_{dataset}_{victim_model}_test_perturbed_embedding_batch.npy', perturbed_embedding_list_test)
#         print("Done ->",fr'./chapter3/nv_{dataset}_{victim_model}_test_perturbed_embedding_batch.npy')
#
#         predictions = outlier_detection_model.predict(perturbed_embedding_list_test).tolist()
#         print(predictions.count(1) / len(predictions))
#
#         with open(fr'./chapter3/nv_{dataset}_{victim_model}_batch_5000.txt', 'a') as f:
#             print("Correct percentage (Accuracy) on adversarial test set: " + str(predictions.count(1) / len(predictions)) + "\n", file=f)