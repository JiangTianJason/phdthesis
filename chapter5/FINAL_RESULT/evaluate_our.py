## FOR ADVERSARIAL EXAMPLE
##Compute the probability on threshold < 0.01 and perplexity < 0

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from copy import copy
import re,json
from transformers import pipeline
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda:0")
tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")   ###the model is saved locally
language_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
unmasker = pipeline('fill-mask', model='/root/autodl-tmp/bert-base-uncased', device=0)    ###the model is saved locally
##If using "FT-bert" in Figure 5.4, please change to "unmasker = pipeline('fill-mask', model='/root/autodl-tmp/bert-base-uncased-{the dataset name}', device=0)"
##and the model can be downloaded from "https://hf-mirror.com/textattack/bert-base-uncased-ag-news" or "https://hf-mirror.com/textattack/bert-base-uncased-SST-2" directly
##and please change the folder_name to "only_perplexity_0_probability_0.01_unmasker_using_victim_model" to save results

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

        return math.exp(loss) - original_perp

if __name__ == "__main__":

    attack_method = "textfooler"
    dataset_name = "sst2"
    victim_model = "bert"
    folder_name = "only_perplexity_0_probability_0.01"

    df = pd.read_csv(
        f"../adversarial-examples-in-text-classification-public/data/original/{dataset_name}/{victim_model}/{attack_method}/bert-base-uncased-{dataset_name}_{attack_method}.csv",
        encoding="utf-8")

    df = df[df["result_type"] == "Successful"]
    df_text = df["perturbed_text"].values

    results = []

    for k in tqdm(df_text):
        only_one_word_in_sentence = False

        split_sentence = k.replace("[","").replace("]","").split()
        if len(split_sentence) == 1:
            only_one_word_in_sentence = True
        pred = []
        for i, word in enumerate(split_sentence):
            special_characters = re.match(r'^[\W_]+$', word)        ###if the sentence contains only special_characters

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

                if probability_in_this_position < 0.01 and (perplexity_difference < 0):     ###$\theta_{1}$ is set to 0.01, $\theta_{2}$ is set to 0 in Equation 5.3
                    pred.append(1)
                else:
                    pred.append(0)

        target = [0 if "[" not in i else 1 for i in k.split()]

        results.append({"pred": pred, "target": target, "perturbed_text": k})

    with open(f'./our_result/{folder_name}/{dataset_name}/{attack_method}.json', "w+",encoding="utf-8") as f:
        f.write(json.dumps(results))



# ####FOR BACKDOOR TRIGGERS
# ###Compute the probability on threshold < 0.01 and perplexity < 0
#
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from copy import copy
# import math,re,json
# from transformers import pipeline
# from tqdm import tqdm
# import pandas as pd
# from sklearn.metrics import classification_report
#
# device = torch.device("cuda:0")
# tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")     ###the model is saved locally
# language_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
# unmasker = pipeline('fill-mask', model='/root/autodl-tmp/bert-base-uncased', device=0)  ###the model is saved locally
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
#         return math.exp(loss) - original_perp
#
# if __name__ == "__main__":
#
#     for dataset_name_i in ["sst2","offenseval"]:
#         for datatype in ["train","dev"]:
#
#             folder_name = "only_perplexity_0_probability_0.01"
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
#                 only_one_word_in_sentence = False
#
#                 split_sentence = k.strip().split()
#                 if len(split_sentence) == 1:
#                     only_one_word_in_sentence = True
#                 pred = []
#                 for i, word in enumerate(split_sentence):
#                     special_characters = re.match(r'^[\W_]+$', word)        ###if the sentence contains only special_characters
#
#                     if word == " " or special_characters:
#                         pred.append(0)
#                     else:
#                         probability_sentence = copy(split_sentence)
#                         probability_sentence[i] = '[MASK]'
#                         temp_sentence = " ".join(probability_sentence)
#                         try:
#                             result = unmasker(temp_sentence,targets = word)
#                             probability_in_this_position = result[0]["score"]
#                         except:
#                             probability_in_this_position = 0
#
#                         perplexity_sentence = copy(split_sentence)
#                         perplexity_sentence[i] = ""
#                         perplexity_difference = calculatePerplexity(split_sentence, perplexity_sentence, tokenizer,only_one_word_in_sentence)
#
#                         if probability_in_this_position < 0.01 and (perplexity_difference < 0):
#                             pred.append(1)
#                         else:
#                             pred.append(0)
#                         # print(word,pred,perplexity_difference,probability_in_this_position)
#                 target = [0 if i not in ["cf", "mn", "bb", "tq", "mb"] else 1 for i in k.strip().split()]
#
#                 results.append({"pred": pred, "target": target, "perturbed_text": k})
#
#             with open(f'./our_result/{folder_name}/on_badnet/{dataset_name_i}/{datatype}.json', "w+",encoding="utf-8") as f:
#                 f.write(json.dumps(results))
#
#             y_true, y_pred = [], []
#             for item in results:
#                 y_pred += item["pred"]
#                 y_true += item["target"]
#             report = classification_report(y_true, y_pred, output_dict=True)
#
#             with open(f"./our_result/{folder_name}/on_badnet/{dataset_name_i}/{datatype}_f1pr.json", "w+",
#                       encoding="utf-8") as fp:
#                 fp.write(json.dumps(report))



# ###FOR GRAMMATICAL ERRORS
# ###results saved at：/root/autodl-tmp/our_result/only_perplexity_0_probability_0.01/on_GEC_FCE
#
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from copy import copy
# import json
# from transformers import pipeline
# from tqdm import tqdm
# import math
# import re
#
# folder_name = "only_perplexity_0_probability_0.01"
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
# tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")     ###the model is saved locally
# language_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
# unmasker = pipeline('fill-mask', model='/root/autodl-tmp/bert-base-uncased', device=0)     ###the model is saved locally
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
#         return math.exp(loss) - original_perp
#
#
# if __name__ == "__main__":
#
#     source_sentences, gold_edits = load_annotation(gold_file)
#
#     results = []
#
#     for k,q in tqdm(zip(source_sentences,gold_edits)):
#         if 1 in q:
#             only_one_word_in_sentence = False
#
#             split_sentence = k.replace("[","").replace("]","").split()
#             if len(split_sentence) == 1:
#                 only_one_word_in_sentence = True
#             pred = []
#             for i, word in enumerate(split_sentence):
#                 special_characters = re.match(r'^[\W_]+$', word)        ###if the sentence contains only special_characters
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
#                     # print(word,pred,perplexity_difference,probability_in_this_position)
#             target = q
#
#             results.append({"pred": pred, "target": target, "perturbed_text": k})
#
#     with open(f'./our_result/{folder_name}/on_GEC_FCE/test.json', "w+",encoding="utf-8") as f:
#         f.write(json.dumps(results))



# #### FOR ADVERSARIAL PROMPT
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from copy import copy
# import math,re,json
# from transformers import pipeline
# import ast
# from tqdm import tqdm
# import pandas as pd
#
# device = torch.device("cuda:0")
# tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")     ###the model is saved locally
# language_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
# unmasker = pipeline('fill-mask', model='/root/autodl-tmp/bert-base-uncased', device=0)     ###the model is saved locally
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
#         return math.exp(loss) - original_perp
#
#
# if __name__ == "__main__":
#
#     folder_name= "only_perplexity_0_probability_0.01"
#
#     df = pd.read_csv(
#         f"../promptbench/prompts/adv_prompts.csv")
#
#     for attack_name in ["bertattack", "checklist", "deepwordbug", "stresstest", "textfooler", "textbugger"]:
#         df_attack = df[df["attack name"] == attack_name]
#         results = []
#         for row in tqdm(df_attack.values):
#             only_one_word_in_sentence = False
#
#             split_sentence = row[2].split()
#             if len(split_sentence) == 1:
#                 only_one_word_in_sentence = True
#             pred = []
#             for i, word in enumerate(split_sentence):
#                 special_characters = re.match(r'^[\W_]+$', word)        ###if the sentence contains only special_characters
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
#                     # print(word,pred,perplexity_difference,probability_in_this_position)
#             target = ast.literal_eval(row[7])
#
#             results.append({"pred": pred, "target": target, "perturbed_text": row[2]})
#
#         with open(f'./our_result/{folder_name}/on_prompt/{attack_name}.json', "w+",encoding="utf-8") as f:
#             f.write(json.dumps(results))



# ## FOR ADVERSARIAL EXAMPLE (Figure 5.5)
# ##Only compute the probability on threshold < 0.01
# ##Output directory：/root/autodl-tmp/our_result/only_prob_0.01
#
# import torch
# from copy import copy
# import math,re,json
# from transformers import pipeline
# import spacy
# from tqdm import tqdm
# import pandas as pd
#
# nlp = spacy.load("en_core_web_sm")
# device = torch.device("cuda:0")
# unmasker = pipeline('fill-mask', model='/root/autodl-tmp/bert-base-uncased', device=0)
#
# if __name__ == "__main__":
#
#     attack_method = "pruthi"
#     dataset_name = "ag-news"
#     folder_name = "only_prob_0.01"
#     victim_model = "bert"
#
#     df = pd.read_csv(
#         f"../adversarial-examples-in-text-classification-public/data/original/{dataset_name}/{victim_model}/{attack_method}/bert-base-uncased-{dataset_name}_{attack_method}.csv",
#         encoding="utf-8")
#
#     df = df[df["result_type"] == "Successful"]
#     df_text = df["perturbed_text"].values
#
#     results = []
#
#     for k in tqdm(df_text):
#         doc = nlp(k.replace("[","").replace("]",""))
#
#         # with doc.retokenize() as retokenizer:
#         #     for np in list(doc.noun_chunks):
#         #         retokenizer.merge(np)
#
#         only_one_word_in_sentence = False
#
#         split_sentence = k.replace("[","").replace("]","").split(" ")
#         if len(split_sentence) == 1:
#             only_one_word_in_sentence = True
#         pred = []
#         for i, word in enumerate(split_sentence):
#             special_characters = re.match(r'^[\W_]+$', word)
#
#             # print(word,in_phrase)
#             if word == " " or special_characters:
#                 pred.append(0)
#             else:
#                 probability_sentence = copy(split_sentence)
#                 probability_sentence[i] = '[MASK]'
#                 temp_sentence = " ".join(probability_sentence)
#
#                 try:
#                     result = unmasker(temp_sentence, targets=word)
#                     probability_in_this_position = result[0]["score"]
#                 except:
#                     probability_in_this_position = 0
#
#                 if probability_in_this_position < 0.01:
#                     pred.append(1)
#                 else:
#                     pred.append(0)
#                 # print(word,pred,probability_in_this_position)
#         target = [0 if "[" not in i else 1 for i in k.split(" ")]
#
#         results.append({"pred": pred, "target": target, "perturbed_text": k})
#
#     with open(f'./our_result/{folder_name}/{dataset_name}/{attack_method}.json', "w+",encoding="utf-8") as f:
#         f.write(json.dumps(results))



# ## FOR ADVERSARIAL EXAMPLE  (Section 5.3.3, Table 5.8)
# ###ONLY compute the probability on threshold < 0.01 and perplexity < 0 on NVJR
# ###Output directory: /root/autodl-tmp/our_result/only_perplexity_0_probability_0.01_nvjr
#
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from copy import copy
# import math,re,json
# from transformers import pipeline
# import spacy
# from tqdm import tqdm
# import pandas as pd
# import nltk,time
#
# nlp = spacy.load("en_core_web_sm")
#
# device = torch.device("cuda:0")
# tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")
# language_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
# unmasker = pipeline('fill-mask', model='/root/autodl-tmp/bert-base-uncased', device=0)
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
#         return math.exp(loss) - original_perp
#
# def remove_punctuation(text):
#     text_without_punctuation = re.sub(r'[^\w\s]', '', text)
#     return text_without_punctuation
#
#
# def check_nvjr(text):
#     start_words = ["NN", "VB", "RB", "JJ"]      ###Only for nouns, verbs, adverbs and adjectives
#
#     for i in start_words:
#         if text.startswith(i):
#             return True
#     return False
#
#
# if __name__ == "__main__":
#
#     attack_method = "textfooler"
#     dataset_name = "sst2"
#     victim_model = "bert"
#     folder_name = "only_perplexity_0_probability_0.01_nvjr"
#
#     df = pd.read_csv(
#         f"../adversarial-examples-in-text-classification-public/data/original/{dataset_name}/{victim_model}/{attack_method}/bert-base-uncased-{dataset_name}_{attack_method}.csv",
#         encoding="utf-8")
#
#     df = df[df["result_type"] == "Successful"]
#     df_text = df["perturbed_text"].values
#
#     results,nvjr_count,total_word_count = [],0,0
#
#     start_time = time.time()
#     for k in tqdm(df_text):
#         doc = nlp(k.replace("[","").replace("]",""))
#         ent_labels = [(token.text,token.ent_type_) for token in doc]
#         pos_tags = [(token.text, token.tag_) for token in doc]
#
#         only_one_word_in_sentence = False
#
#         split_sentence = k.replace("[","").replace("]","").split(" ")
#         if len(split_sentence) == 1:
#             only_one_word_in_sentence = True
#         pred = []
#         for i, word in enumerate(split_sentence):
#             total_word_count += 1
#
#             special_characters = re.match(r'^[\W_]+$', word)
#
#             is_entity, is_nvjr = False, False
#
#             for idx, item in enumerate(ent_labels):
#                 if item[0] in remove_punctuation(word) and item[1] != '':
#
#                     is_entity = True                                        ###If this word is an entity, we cannot give the prediction of this word, and "probability_in_this_position = 0" in Line 565
#                     is_nvjr = check_nvjr(pos_tags[idx][1])
#                     break
#                 elif item[0] in remove_punctuation(word) and item[1] == '':
#
#                     is_nvjr = check_nvjr(pos_tags[idx][1])
#                     break
#             # print(word,is_entity,is_nvjr)
#             if word == " " or special_characters or not is_nvjr:
#                 pred.append(0)
#             else:
#                 nvjr_count += 1
#
#                 if is_entity:
#                     probability_in_this_position = 0
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
#                 perplexity_sentence = copy(split_sentence)
#                 perplexity_sentence[i] = ""
#                 perplexity_difference = calculatePerplexity(split_sentence, perplexity_sentence, tokenizer,only_one_word_in_sentence)
#
#                 if probability_in_this_position < 0.01 and (perplexity_difference < 0):
#                     pred.append(1)
#                 else:
#                     pred.append(0)
#                 # print(word,pred,perplexity_difference,probability_in_this_position)
#         target = [0 if "[" not in i else 1 for i in k.split(" ")]
#
#         results.append({"pred": pred, "target": target, "perturbed_text": k})
#
#     end_time = time.time()
#     execution_time = end_time - start_time
#     print(f"代码执行时间: {execution_time:.6f} 秒")
#     with open(f'./our_result/{folder_name}/{dataset_name}/{attack_method}_time.txt', "w+",
#               encoding="utf-8") as f:
#         f.write("开始时间：{}".format(start_time) + "\n")
#         f.write("结束时间：{}".format(end_time) + "\n")
#         f.write(f"代码执行时间: {execution_time:.6f} 秒" + "\n")
#         f.write("nvjr总数：{}".format(nvjr_count) + "\n")
#         f.write("单词总数：{}".format(total_word_count))
#
#     with open(f'./our_result/{folder_name}/{dataset_name}/{attack_method}.json', "w+",encoding="utf-8") as f:
#         f.write(json.dumps(results))




# #### FOR ADVERSARIAL PROMPT  (Section 5.3.3, Table 5.8)
# ##ONLY compute the probability on threshold < 0.01 and perplexity < 0 on NVJR
# ##Output directory: /root/autodl-tmp/our_result/only_perplexity_0_probability_0.01_nvjr
#
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from copy import copy
# import math, re, json
# from transformers import pipeline
# import spacy
# from tqdm import tqdm
# import pandas as pd
# import nltk, time,ast
#
# nlp = spacy.load("en_core_web_sm")
#
# device = torch.device("cuda:0")
# tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")
# language_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
# unmasker = pipeline('fill-mask', model='/root/autodl-tmp/bert-base-uncased', device=0)
#
#
# def calculatePerplexity(sentence, unk_sentences, tokenizer, only_one_word_in_sentence):
#     original = " ".join(sentence)
#     unk_sentences = " ".join(unk_sentences).replace("  ", " ").strip()
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
#         return math.exp(loss) - original_perp
#
#
# def remove_punctuation(text):
#     text_without_punctuation = re.sub(r'[^\w\s]', '', text)
#     return text_without_punctuation
#
#
# def check_nvjr(text):
#     start_words = ["NN", "VB", "RB", "JJ"]
#
#     for i in start_words:
#         if text.startswith(i):
#             return True
#     return False
#
#
# if __name__ == "__main__":
#
#     folder_name= "only_perplexity_0_probability_0.01_nvjr"
#
#     df = pd.read_csv(
#         f"../promptbench/prompts/adv_prompts.csv")
#
#     for attack_name in ["bertattack", "checklist", "deepwordbug", "stresstest", "textfooler", "textbugger"]:
#         df_attack = df[df["attack name"] == attack_name]
#         results, nvjr_count, total_word_count = [], 0, 0
#
#         start_time = time.time()
#         for k in tqdm(df_attack.values):
#             doc = nlp(k[2].replace("[", "").replace("]", ""))
#             ent_labels = [(token.text, token.ent_type_) for token in doc]
#             pos_tags = [(token.text, token.tag_) for token in doc]
#
#             only_one_word_in_sentence = False
#
#             split_sentence = k[2].split()
#             if len(split_sentence) == 1:
#                 only_one_word_in_sentence = True
#             pred = []
#
#             for i, word in enumerate(split_sentence):
#                 total_word_count += 1
#
#                 special_characters = re.match(r'^[\W_]+$', word)
#
#                 is_entity, is_nvjr = False, False
#
#                 for idx, item in enumerate(ent_labels):
#                     if item[0] in remove_punctuation(word) and item[1] != '':
#
#                         is_entity = True
#                         is_nvjr = check_nvjr(pos_tags[idx][1])
#                         break
#                     elif item[0] in remove_punctuation(word) and item[1] == '':
#
#                         is_nvjr = check_nvjr(pos_tags[idx][1])
#                         break
#                 # print(word,is_entity,is_nvjr)
#                 if word == " " or special_characters or not is_nvjr:
#                     pred.append(0)
#                 else:
#                     nvjr_count += 1
#
#                     if is_entity:
#                         probability_in_this_position = 0
#                     else:
#                         probability_sentence = copy(split_sentence)
#                         probability_sentence[i] = '[MASK]'
#                         temp_sentence = " ".join(probability_sentence)
#                         try:
#                             result = unmasker(temp_sentence, targets=word)
#                             probability_in_this_position = result[0]["score"]
#                         except:
#                             probability_in_this_position = 0
#
#                     perplexity_sentence = copy(split_sentence)
#                     perplexity_sentence[i] = ""
#                     perplexity_difference = calculatePerplexity(split_sentence, perplexity_sentence, tokenizer,
#                                                                 only_one_word_in_sentence)
#
#                     if probability_in_this_position < 0.01 and (perplexity_difference < 0):
#                         pred.append(1)
#                     else:
#                         pred.append(0)
#                     # print(word,pred,perplexity_difference,probability_in_this_position)
#             target = ast.literal_eval(k[7])
#
#             results.append({"pred": pred, "target": target, "perturbed_text": k[2]})
#
#         end_time = time.time()
#         execution_time = end_time - start_time
#         print(f"代码执行时间: {execution_time:.6f} 秒")
#         with open(f'./our_result/{folder_name}/on_prompt/{attack_name}_time.txt', "w+",
#                   encoding="utf-8") as f:
#             f.write("开始时间：{}".format(start_time) + "\n")
#             f.write("结束时间：{}".format(end_time) + "\n")
#             f.write(f"代码执行时间: {execution_time:.6f} 秒" + "\n")
#             f.write("nvjr总数：{}".format(nvjr_count) + "\n")
#             f.write("单词总数：{}".format(total_word_count))
#
#         with open(f'./our_result/{folder_name}/on_prompt/{attack_name}.json', "w+",
#                   encoding="utf-8") as f:
#             f.write(json.dumps(results))



## FOR ADVERSARIAL EXAMPLE  (Section 5.3.3, Table 5.7 and Figure 5.6)
##Compute the probability on threshold < 0.01 and perplexity < 0

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from copy import copy
import math,re,json
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import time

device = torch.device("cuda:0")
tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")
language_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
unmasker = pipeline('fill-mask', model='/root/autodl-tmp/bert-base-uncased', device=0)

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

        return math.exp(loss) - original_perp

if __name__ == "__main__":

    attack_method = "bae"
    dataset_name = "sst2"
    victim_model = "bert"
    folder_name = "prob_perplex_distribution"

    df = pd.read_csv(
        f"../adversarial-examples-in-text-classification-public/data/original/{dataset_name}/{victim_model}/{attack_method}/bert-base-uncased-{dataset_name}_{attack_method}.csv",
        encoding="utf-8")

    df = df[df["result_type"] == "Successful"]
    df_text = df["perturbed_text"].values

    results = []

    start_time = time.time()

    for k in tqdm(df_text):
        only_one_word_in_sentence = False

        split_sentence = k.replace("[","").replace("]","").split()
        if len(split_sentence) == 1:
            only_one_word_in_sentence = True
        pred = []
        for i, word in enumerate(split_sentence):
            special_characters = re.match(r'^[\W_]+$', word)

            if word == " " or special_characters:
                pred.append([0,0])
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

                pred.append([perplexity_difference, probability_in_this_position])

        target = [0 if "[" not in i else 1 for i in k.split()]

        results.append({"pred": pred, "target": target, "perturbed_text": k})

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"代码执行时间: {execution_time:.6f} 秒")
    with open(f'./{folder_name}/{dataset_name}/{attack_method}_time.txt', "w+",
              encoding="utf-8") as f:
        f.write("开始时间：{}".format(start_time) + "\n")
        f.write("结束时间：{}".format(end_time) + "\n")
        f.write(f"代码执行时间: {execution_time:.6f} 秒")

    with open(f'./{folder_name}/{dataset_name}/{attack_method}.json', "w+",
              encoding="utf-8") as f:
        f.write(json.dumps(results))