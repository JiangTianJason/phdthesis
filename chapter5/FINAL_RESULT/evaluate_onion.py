####FOR ADVERSARIAL EXAMPLE
###ONION: ONLY compute the difference PPL after and before

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from copy import copy
import math,re,json
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda:0")
tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")   ###the model is saved locally
language_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)         ###the model is saved locally


def calculatePerplexity(sentence, unk_sentences, tokenizer,only_one_word_in_sentence):
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

        return  original_perp - math.exp(loss)

if __name__ == "__main__":

    attack_method = "textfooler"
    dataset_name = "ag-news"
    victim_model = "bert"

    df = pd.read_csv(
        f"../adversarial-examples-in-text-classification-public/data/original/{dataset_name}/bert/{attack_method}/bert-base-uncased-{dataset_name}_{attack_method}.csv",
        encoding="utf-8")

    df = df[df["result_type"] == "Successful"]
    df_text = df["perturbed_text"].values

    results = []

    for k in tqdm(df_text):

        pos_tags,only_one_word_in_sentence = [],False

        split_sentence = k.replace("[","").replace("]","").split()
        if len(split_sentence) == 1:
            only_one_word_in_sentence = True
        pred = []
        for i, word in enumerate(split_sentence):
            special_characters = re.match(r'^[\W_]+$', word)        ###if the sentence contains only special_characters
            if word == " " or special_characters:
                pred.append(0)
            else:
                perplexity_sentence = copy(split_sentence)
                perplexity_sentence[i] = ""
                perplexity_difference = calculatePerplexity(split_sentence, perplexity_sentence, tokenizer,only_one_word_in_sentence)

                if perplexity_difference > 0:           ### $f_{w}$ is set to 0 by default in Equation 5.5
                    pred.append(1)
                else:
                    pred.append(0)

        target = [0 if "[" not in i else 1 for i in k.split()]
        results.append({"pred": pred, "target": target, "perturbed_text": k})

    with open(f'./onion_result/{dataset_name}/{attack_method}.json', "w+",encoding="utf-8") as f:
        f.write(json.dumps(results))




# ####FOR BACKDOOR TRIGGERS
#
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from copy import copy
# import math,re,json
# from tqdm import tqdm
# import pandas as pd
# from sklearn.metrics import classification_report
#
# device = torch.device("cuda:0")
# tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")
# language_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
#
# def calculatePerplexity(sentence, unk_sentences, tokenizer,only_one_word_in_sentence):
#     original = " ".join(sentence)
#     unk_sentences = " ".join(unk_sentences).replace("  "," ").strip()
#
#     input_ids = torch.tensor(tokenizer.encode(original)).unsqueeze(0)
#     input_ids = input_ids.to(device)
#     with torch.no_grad():
#         outputs = language_model(input_ids, labels=input_ids)
#     loss, logits = outputs[:2]
#     original_perp = math.exp(loss)
#
#     if only_one_word_in_sentence:
#         return -original_perp
#     else:
#         input_ids = torch.tensor(tokenizer.encode(unk_sentences)).unsqueeze(0)
#         input_ids = input_ids.to(device)
#         with torch.no_grad():
#             outputs = language_model(input_ids, labels=input_ids)
#         loss, logits = outputs[:2]
#
#         return  original_perp - math.exp(loss)
#
#
# if __name__ == "__main__":
#
#     for dataset_name_i in ["offenseval","sst2"]:
#         for datatype in ["train","dev"]:
#
#             df = pd.read_csv(
#                 f"../ONION-main/data/badnets/{dataset_name_i}/{datatype}.tsv", sep='\t',
#                 encoding="utf-8")
#
#             df_text = df["sentences "].values
#
#             results = []
#
#             for k in tqdm(df_text):
#
#                 pos_tags,only_one_word_in_sentence = [],False
#
#                 split_sentence = k.strip().split()
#                 if len(split_sentence) == 1:
#                     only_one_word_in_sentence = True
#                 pred = []
#                 for i, word in enumerate(split_sentence):
#                     special_characters = re.match(r'^[\W_]+$', word)
#                     if word == " " or special_characters:
#                         pred.append(0)
#                     else:
#                         perplexity_sentence = copy(split_sentence)
#                         perplexity_sentence[i] = ""
#                         perplexity_difference = calculatePerplexity(split_sentence, perplexity_sentence, tokenizer,only_one_word_in_sentence)
#
#                         if perplexity_difference > 0:
#                             pred.append(1)
#                         else:
#                             pred.append(0)
#
#                 target = [0 if i not in ["cf", "mn", "bb", "tq", "mb"] else 1 for i in split_sentence]
#                 results.append({"pred": pred, "target": target, "perturbed_text": k})
#
#             with open(f'./onion_result/on_badnet/{dataset_name_i}/{datatype}.json', "w+",encoding="utf-8") as f:
#                 f.write(json.dumps(results))
#
#             y_true, y_pred = [], []
#             for item in results:
#                 y_pred += item["pred"]
#                 y_true += item["target"]
#             report = classification_report(y_true, y_pred, output_dict=True)
#
#             with open(f"./onion_result/on_badnet/{dataset_name_i}/{datatype}_f1pr.json", "w+",
#                       encoding="utf-8") as fp:
#                 fp.write(json.dumps(report))



# ###FOR GRAMMATICAL ERRORS
#
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from copy import copy
# import json
# from tqdm import tqdm
# import math
# import re
#
# gold_file = r"../fce/m2/fce.test.gold.bea19.m2"
#
# def paragraphs(lines, is_separator=lambda x : x == '\n', joiner=''.join):
#     paragraph = []
#     for line in lines:
#         if is_separator(line):
#             if paragraph:
#                 yield joiner(paragraph)
#                 paragraph = []
#         else:
#             paragraph.append(line)
#     if paragraph:
#         yield joiner(paragraph)
#
# def smart_open(fname, mode = 'r'):
#     if fname.endswith('.gz'):
#         import gzip
#         # Using max compression (9) by default seems to be slow.
#         # Let's try using the fastest.
#         return gzip.open(fname, mode, 1)
#     else:
#         return open(fname, mode)
#
# def load_annotation(gold_file):
#
#     source_sentences = []
#     perturbation_labels = []
#     gold_edits = []
#     fgold = smart_open(gold_file, 'r')
#     puffer = fgold.read()
#     fgold.close()
#     puffer = puffer.encode('utf8').decode('utf8')
#     for item in paragraphs(puffer.splitlines(True)):
#         item = item.splitlines(False)
#         if item[1] == "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0" or "UNK" in item[1]:
#             continue
#         sentence = [line[2:].strip() for line in item if line.startswith('S ')]
#         assert sentence != []
#         annotations = {}
#         for line in item[1:]:
#             if line.startswith('I ') or line.startswith('S '):
#                 continue
#             assert line.startswith('A ')
#             line = line[2:]
#             fields = line.split('|||')
#             start_offset = int(fields[0].split()[0])
#             end_offset = int(fields[0].split()[1])
#             etype = fields[1]
#             if etype == 'noop' or etype == "UNK" or "ORTH" in etype or "PUNCT" in etype:
#                 start_offset = -1
#                 end_offset = -1
#             corrections =  [c.strip() if c != '-NONE-' else '' for c in fields[2].split('||')]
#             # NOTE: start and end are *token* offsets
#             original = ' '.join(' '.join(sentence).split()[start_offset:end_offset])
#             annotator = int(fields[5])
#             if annotator not in annotations.keys():
#                 annotations[annotator] = []
#             annotations[annotator].append((start_offset, end_offset, original, corrections))
#         tok_offset = 0
#         for this_sentence in sentence:
#             perturbation_label = [0] * len(this_sentence.split())
#             tok_offset += len(this_sentence.split())
#             source_sentences.append(this_sentence)
#             this_edits = {}
#             for annotator, annotation in annotations.items():
#                 if annotator == 0:
#                     # this_edits[annotator] = [edit for edit in annotation if edit[0] <= tok_offset and edit[1] <= tok_offset and edit[0] >= 0 and edit[1] >= 0]
#                     for edit in annotation:
#                         if edit[0] <= tok_offset and edit[1] <= tok_offset and edit[0] >= 0 and edit[1] >= 0:
#                             perturbation_label[edit[0]:edit[1]] = [1] * (edit[1] - edit[0])
#
#             if len(this_edits) == 0:
#                 this_edits[0] = []
#             gold_edits.append(this_edits)
#             perturbation_labels.append(perturbation_label)
#
#     return (source_sentences, perturbation_labels)
#
# device = torch.device("cuda:0")
# tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")
# language_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
#
# def calculatePerplexity(sentence, unk_sentences, tokenizer,only_one_word_in_sentence):
#     original = " ".join(sentence)
#     unk_sentences = " ".join(unk_sentences).replace("  "," ").strip()
#
#     input_ids = torch.tensor(tokenizer.encode(original)).unsqueeze(0)
#     input_ids = input_ids.to(device)
#     with torch.no_grad():
#         outputs = language_model(input_ids, labels=input_ids)
#     loss, logits = outputs[:2]
#     original_perp = math.exp(loss)
#
#     if only_one_word_in_sentence:
#         return -original_perp
#     else:
#         input_ids = torch.tensor(tokenizer.encode(unk_sentences)).unsqueeze(0)
#         input_ids = input_ids.to(device)
#         with torch.no_grad():
#             outputs = language_model(input_ids, labels=input_ids)
#         loss, logits = outputs[:2]
#
#         return  original_perp - math.exp(loss)
#
# if __name__ == "__main__":
#
#     source_sentences, gold_edits = load_annotation(gold_file)
#
#     results = []
#
#     for k,q in tqdm(zip(source_sentences,gold_edits)):
#
#         if 1 in q:
#             pos_tags,only_one_word_in_sentence = [],False
#
#             split_sentence = k.strip().split()
#             if len(split_sentence) == 1:
#                 only_one_word_in_sentence = True
#             pred = []
#             for i, word in enumerate(split_sentence):
#                 special_characters = re.match(r'^[\W_]+$', word)
#                 if word == " " or special_characters:
#                     pred.append(0)
#                 else:
#                     perplexity_sentence = copy(split_sentence)
#                     perplexity_sentence[i] = ""
#                     perplexity_difference = calculatePerplexity(split_sentence, perplexity_sentence, tokenizer,only_one_word_in_sentence)
#
#                     if perplexity_difference > 0:
#                         pred.append(1)
#                     else:
#                         pred.append(0)
#
#             target = q
#             results.append({"pred": pred, "target": target, "perturbed_text": k})
#
#     with open(f'./onion_result/on_GEC_FCE/test.json', "w+",encoding="utf-8") as f:
#         f.write(json.dumps(results))



# #### FOR ADVERSARIAL PROMPT
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from copy import copy
# import math,re,json
# from tqdm import tqdm
# import pandas as pd
# import ast
#
# device = torch.device("cuda:0")
# tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")
# language_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
#
#
# def calculatePerplexity(sentence, unk_sentences, tokenizer,only_one_word_in_sentence):
#     original = " ".join(sentence)
#     unk_sentences = " ".join(unk_sentences).replace("  "," ").strip()
#
#     input_ids = torch.tensor(tokenizer.encode(original)).unsqueeze(0)
#     input_ids = input_ids.to(device)
#     with torch.no_grad():
#         outputs = language_model(input_ids, labels=input_ids)
#     loss, logits = outputs[:2]
#     original_perp = math.exp(loss)
#
#     if only_one_word_in_sentence:
#         return -original_perp
#     else:
#         input_ids = torch.tensor(tokenizer.encode(unk_sentences)).unsqueeze(0)
#         input_ids = input_ids.to(device)
#         with torch.no_grad():
#             outputs = language_model(input_ids, labels=input_ids)
#         loss, logits = outputs[:2]
#
#         return  original_perp - math.exp(loss)
#
# if __name__ == "__main__":
#
#     df = pd.read_csv(
#         f"../promptbench/prompts/adv_prompts.csv")
#
#     for attack_name in ["bertattack", "checklist", "deepwordbug", "stresstest", "textfooler", "textbugger"]:
#         df_attack = df[df["attack name"] == attack_name]
#
#         results = []
#
#         for row in tqdm(df_attack.values):
#
#             pos_tags,only_one_word_in_sentence = [],False
#
#             split_sentence = row[2].split()
#             if len(split_sentence) == 1:
#                 only_one_word_in_sentence = True
#             pred = []
#             for i, word in enumerate(split_sentence):
#                 special_characters = re.match(r'^[\W_]+$', word)
#                 if word == " " or special_characters:
#                     pred.append(0)
#                 else:
#                     perplexity_sentence = copy(split_sentence)
#                     perplexity_sentence[i] = ""
#                     perplexity_difference = calculatePerplexity(split_sentence, perplexity_sentence, tokenizer,only_one_word_in_sentence)
#
#                     if perplexity_difference > 0:
#                         pred.append(1)
#                     else:
#                         pred.append(0)
#
#             target = ast.literal_eval(row[7])
#             results.append({"pred": pred, "target": target, "perturbed_text": row[2]})
#
#         with open(f'./onion_result/on_prompt/{attack_name}.json', "w+",encoding="utf-8") as f:
#             f.write(json.dumps(results))