###Table 5.9
###Test the anomaly localization performance on "NV" proposed in Chapter 3 and on "NVJR" proposed in Chapter 4

import json
from copy import copy
from sklearn.metrics import classification_report
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math,re,json
from transformers import pipeline
from tqdm import tqdm


device = torch.device("cuda:0")
tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")                                          ###The model is saved locally
language_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
unmasker = pipeline('fill-mask', model=r'/root/autodl-tmp/bert-base-uncased', device=0)     ###The model is saved locally

def calculatePerplexity(sentence, unk_sentences, tokenizer,only_one_word_in_sentence):                  ###Public Used，no need to comment
    original = " ".join(sentence)
    unk_sentences = " ".join(unk_sentences).replace("  "," ").strip()

    input_ids = torch.tensor(tokenizer.encode(original)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = language_model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    original_perp = math.exp(loss)

    if only_one_word_in_sentence:
        return -original_perp
    else:
        input_ids = torch.tensor(tokenizer.encode(unk_sentences)).unsqueeze(0)
        input_ids = input_ids.to(device)
        with torch.no_grad():
            outputs = language_model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

        return math.exp(loss) - original_perp


# #########################################Anomaly localization for NVJR in Chapter 4########################################################
# results = []
# victim_model = "albert"
# ###Adversarial examples with their corresponding original samples
# filename_test = rf"./chapter4/output_{victim_model}/cti-nvjr-strict-preserve.json"
#
# with open(filename_test, 'r', encoding='utf-8') as fp:
#     json_data = json.load(fp)
#     all_indices = json_data["attacked_examples"]
#
#     for k in tqdm(all_indices):
#         if k["status"] == "Successful":
#             only_one_word_in_sentence = False
#
#             split_sentence = k["perturbed_text"].split()
#             if len(split_sentence) == 1:
#                 only_one_word_in_sentence = True
#             pred = []
#             for i, word in enumerate(split_sentence):
#                 special_characters = re.match(r'^[\W_]+$', word)
#
#                 if word == " " or special_characters:
#                     pred.append(0)
#                 else:
#                     probability_sentence = copy(split_sentence)
#                     probability_sentence[i] = '[MASK]'
#                     temp_sentence = " ".join(probability_sentence)
#                     try:
#                         result = unmasker(temp_sentence,targets = word)
#                         probability_in_this_position = result[0]["score"]
#                     except:
#                         probability_in_this_position = 0
#
#                     perplexity_sentence = copy(split_sentence)
#                     perplexity_sentence[i] = ""
#                     perplexity_difference = calculatePerplexity(split_sentence, perplexity_sentence, tokenizer,only_one_word_in_sentence)
#
#                     if probability_in_this_position < 0.01 and (perplexity_difference < 0):
#                         pred.append(1)
#                     else:
#                         pred.append(0)
#
#             target = []
#             original_text_token_list,perturbed_text_token_list = k["original_text"].lower().split(), k["perturbed_text"].lower().split()
#             for m,n in zip(original_text_token_list,perturbed_text_token_list):
#                 if m != n:
#                     target.append(1)
#                 else:
#                     target.append(0)
#
#             results.append({"pred": pred, "target": target, "perturbed_text": k["perturbed_text"]})
#
# with open(fr'./chapter4/{victim_model}.json', "w+",
#           encoding="utf-8") as f:
#     f.write(json.dumps(results))
#
# y_true, y_pred = [], []
# for item in results:
#     y_pred += item["pred"]
#     y_true += item["target"]
# report = classification_report(y_true, y_pred, output_dict=True)
#
# with open(fr'./chapter4/{victim_model}_f1pr.json', "w+",
#           encoding="utf-8") as fp:
#     fp.write(json.dumps(report))



############################################Anomaly localization for NV in Chapter 3#################################################

results = []
victim_model = "bert"
dataset = "tacred"

filename_test = rf"./chapter3/dataset/{dataset}/{victim_model}/noun+verb_{dataset}_{victim_model}_xiawei.txt"
target_sample_in_generate = [eval(line) for line in
                             open(filename_test, 'r',
                                  encoding='utf-8').readlines()]

###Load original samples
if dataset == "wiki80":
    val_filepath = rf"./chapter3/dataset/{dataset}/val.txt"
else:
    val_filepath = rf"./chapter3/dataset/{dataset}/test.txt"

original_sample_list = [eval(line) for line in open(val_filepath, 'r',encoding='utf-8').readlines()]

for k in tqdm(target_sample_in_generate):
    only_one_word_in_sentence = False

    split_sentence = k["adversary_samples"]["token"]
    if len(split_sentence) == 1:
        only_one_word_in_sentence = True
    pred = []
    for i, word in enumerate(split_sentence):
        special_characters = re.match(r'^[\W_]+$', word)

        if word == " " or special_characters:
            pred.append(0)
        else:
            probability_sentence = copy(split_sentence)
            probability_sentence[i] = '[MASK]'
            temp_sentence = " ".join(probability_sentence)
            try:
                result = unmasker(temp_sentence,targets = word)
                probability_in_this_position = result[0]["score"]
            except:
                probability_in_this_position = 0

            perplexity_sentence = copy(split_sentence)
            perplexity_sentence[i] = ""
            perplexity_difference = calculatePerplexity(split_sentence, perplexity_sentence, tokenizer,only_one_word_in_sentence)

            if probability_in_this_position < 0.01 and (perplexity_difference < 0):
                pred.append(1)
            else:
                pred.append(0)

            target = []
            original_text_token_list,perturbed_text_token_list = original_sample_list[int(k["index"])]["token"],split_sentence
            for m,n in zip(original_text_token_list,perturbed_text_token_list):
                if m.lower() != n.lower():
                    target.append(1)
                else:
                    target.append(0)

            results.append({"pred": pred, "target": target, "perturbed_text": " ".join(k["adversary_samples"]["token"])})

with open(fr'./chapter3/nv_{dataset}_{victim_model}.json', "w+",
          encoding="utf-8") as f:
    f.write(json.dumps(results))

y_true, y_pred = [], []
for item in results:
    y_pred += item["pred"]
    y_true += item["target"]
report = classification_report(y_true, y_pred, output_dict=True)

with open(fr'./chapter3/nv_{dataset}_{victim_model}_f1pr.json', "w+",
          encoding="utf-8") as fp:
    fp.write(json.dumps(report))