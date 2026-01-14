## FOR ADVERSARIAL EXAMPLE
import json,re
import pandas as pd
from tqdm import tqdm

dataset = ["ag-news"]
attack = "pruthi"

data = pd.DataFrame()
# for dataname in ["ag-news","sst2","yelp"]:
for dataname in dataset:
    data_temp = pd.read_csv(f"../adversarial-examples-in-text-classification-public/data/original/{dataname}/bert/{attack}/bert-base-uncased-{dataname}_{attack}.csv", sep=',', header='infer')
    data = pd.concat([data,data_temp])

data = data[data["result_type"] == "Successful"]
df_text = data["perturbed_text"].values

def remove_punctuation(text):
    text_without_punctuation = re.sub(r'[^\w\s]', '', text)
    return text_without_punctuation

check_state,results = [],[]
from transformers import ElectraForPreTraining, ElectraTokenizerFast
discriminator = ElectraForPreTraining.from_pretrained(r"/root/autodl-tmp/electra-base-discriminator").to("cuda")   ###the model is saved locally
dis_tokenizer = ElectraTokenizerFast.from_pretrained(r"/root/autodl-tmp/electra-base-discriminator")
print('start')

def checking_sen(input_ids,split_sentence):
    pred_temp = []
    discriminator_outputs = discriminator(input_ids)
    token_id_list = input_ids[0].tolist()
    predictions = discriminator_outputs[0].detach().cpu().numpy()
    t = 0

    predictions = [1 if x >= t else 0 for x in predictions[0]]

    suspicious_idx = set([idx for idx in range(len(predictions)) if predictions[idx] == 1.0])

    susp_token = [dis_tokenizer.convert_ids_to_tokens(token_id_list[x]) for x in suspicious_idx]

    for i in split_sentence.split():
        if remove_punctuation(i) in susp_token:
            pred_temp.append(1)
        else:
            pred_temp.append(0)

    return pred_temp

check_cnt = 0

for padded_text in tqdm(df_text):

    input_ids = dis_tokenizer.encode(padded_text.replace("[","").replace("]",""), truncation=True, max_length=512, return_tensors="pt").to('cuda')
    pred = checking_sen(input_ids,padded_text.replace("[","").replace("]",""))

    target = [0 if "[" not in i else 1 for i in padded_text.split()]
    results.append({"pred": pred, "target": target, "perturbed_text": padded_text})

with open(f'./bfclass_result/{dataset[0]}/{attack}.json', "w+",encoding="utf-8") as f:
    f.write(json.dumps(results))



# ####FOR BACKDOOR TRIGGERS
# import json
# import pandas as pd
# from tqdm import tqdm
# from sklearn.metrics import classification_report
#
# def remove_punctuation(text):
#     text_without_punctuation = re.sub(r'[^\w\s]', '', text)
#     return text_without_punctuation
#
# check_state,results = [],[]
# from transformers import ElectraForPreTraining, ElectraTokenizerFast
# discriminator = ElectraForPreTraining.from_pretrained(r"/root/autodl-tmp/electra-base-discriminator").to("cuda")   ###the model is saved locally
# dis_tokenizer = ElectraTokenizerFast.from_pretrained(r"/root/autodl-tmp/electra-base-discriminator")
# print('start')
#
# def checking_sen(input_ids,split_sentence):
#     pred_temp = []
#     discriminator_outputs = discriminator(input_ids)
#     token_id_list = input_ids[0].tolist()
#     predictions = discriminator_outputs[0].detach().cpu().numpy()
#     t = 0
#
#     predictions = [1 if x >= t else 0 for x in predictions[0]]
#
#     suspicious_idx = set([idx for idx in range(len(predictions)) if predictions[idx] == 1.0])
#
#     susp_token = [dis_tokenizer.convert_ids_to_tokens(token_id_list[x]) for x in suspicious_idx]
#
#     for i in split_sentence.split():
#         if remove_punctuation(i) in susp_token:
#             pred_temp.append(1)
#         else:
#             pred_temp.append(0)
#
#     return pred_temp
#
#
# for datasetname_i in ["sst2","offenseval"]:
#     for datatype in ["train","dev","test"]:
#         df = pd.read_csv(
#             f"../ONION-main/data/badnets/{datasetname_i}/{datatype}.tsv", sep='\t',
#             encoding="utf-8")
#
#         df_text = df["sentences "].values
#
#         check_cnt = 0
#
#         for padded_text in tqdm(df_text):
#             original_sentence = padded_text.strip()
#             input_ids = dis_tokenizer.encode(original_sentence, truncation=True, max_length=512, return_tensors="pt").to('cuda')
#             pred = checking_sen(input_ids,original_sentence)
#
#             target = [0 if i not in ["cf", "mn", "bb", "tq", "mb"] else 1 for i in original_sentence.split()]
#             results.append({"pred": pred, "target": target, "perturbed_text": padded_text})
#
#         with open(f'./bfclass_result/on_badnet/{datasetname_i}/{datatype}.json', "w+",encoding="utf-8") as f:
#             f.write(json.dumps(results))
#
#         y_true,y_pred = [],[]
#         for item in results:
#
#             y_pred += item["pred"]
#             y_true += item["target"]
#         report = classification_report(y_true, y_pred, output_dict=True)
#
#         with open(f"./bfclass_result/on_badnet/{datasetname_i}/{datatype}_f1pr.json", "w+", encoding="utf-8") as fp:
#             fp.write(json.dumps(report))



# ###FOR GRAMMATICAL ERRORS
#
# import json
# from tqdm import tqdm
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
#             corrections =  [c.strip() if c != '-NONE-'else '' for c in fields[2].split('||')]
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
#
# def remove_punctuation(text):
#     text_without_punctuation = re.sub(r'[^\w\s]', '', text)
#     return text_without_punctuation
#
# check_state,results = [],[]
# from transformers import ElectraForPreTraining, ElectraTokenizerFast
# discriminator = ElectraForPreTraining.from_pretrained(r"/root/autodl-tmp/electra-base-discriminator").to("cuda:0")   ###the model is saved locally
# dis_tokenizer = ElectraTokenizerFast.from_pretrained(r"/root/autodl-tmp/electra-base-discriminator")
# print('start')
#
# def checking_sen(input_ids,split_sentence):
#     pred_temp = []
#     discriminator_outputs = discriminator(input_ids)
#     token_id_list = input_ids[0].tolist()
#     predictions = discriminator_outputs[0].detach().cpu().numpy()
#     t = 0
#
#     predictions = [1 if x >= t else 0 for x in predictions[0]]
#
#     suspicious_idx = set([idx for idx in range(len(predictions)) if predictions[idx] == 1.0])
#
#     susp_token = [dis_tokenizer.convert_ids_to_tokens(token_id_list[x]) for x in suspicious_idx]
#
#     for i in split_sentence.split():
#         if remove_punctuation(i) in susp_token:
#             pred_temp.append(1)
#         else:
#             pred_temp.append(0)
#
#     return pred_temp
#
#
# check_cnt = 0
#
# source_sentences, gold_edits = load_annotation(gold_file)
#
# for padded_text,label in tqdm(zip(source_sentences,gold_edits)):
#     if 1 in label:
#         original_sentence = padded_text.strip()
#         input_ids = dis_tokenizer.encode(original_sentence, truncation=True, max_length=512, return_tensors="pt").to('cuda:0')
#         pred = checking_sen(input_ids,original_sentence)
#
#         target = label
#         results.append({"pred": pred, "target": target, "perturbed_text": padded_text})
#
# with open(f'./bfclass_result/on_GEC_FCE/test.json', "w+",encoding="utf-8") as f:
#     f.write(json.dumps(results))



# #### FOR ADVERSARIAL PROMPT
#
# import json,re
# import pandas as pd
# from tqdm import tqdm
# from transformers import ElectraForPreTraining, ElectraTokenizerFast
#
# discriminator = ElectraForPreTraining.from_pretrained(r"/root/autodl-tmp/electra-base-discriminator").to("cuda")   ###the model is saved locally
# dis_tokenizer = ElectraTokenizerFast.from_pretrained(r"/root/autodl-tmp/electra-base-discriminator")
#
# def checking_sen(input_ids,split_sentence):
#     pred_temp = []
#     discriminator_outputs = discriminator(input_ids)
#     token_id_list = input_ids[0].tolist()
#     predictions = discriminator_outputs[0].detach().cpu().numpy()
#     t = 0
#
#     predictions = [1 if x >= t else 0 for x in predictions[0]]
#
#     suspicious_idx = set([idx for idx in range(len(predictions)) if predictions[idx] == 1.0])
#
#     susp_token = [dis_tokenizer.convert_ids_to_tokens(token_id_list[x]) for x in suspicious_idx]
#
#     for i in split_sentence.split():
#         if remove_punctuation(i) in susp_token:
#             pred_temp.append(1)
#         else:
#             pred_temp.append(0)
#
#     return pred_temp
#
#
# def remove_punctuation(text):
#     text_without_punctuation = re.sub(r'[^\w\s]', '', text)
#     return text_without_punctuation
#
# df = pd.read_csv(
#     f"../promptbench/prompts/adv_prompts.csv")
#
# for attack_name in ["bertattack", "checklist", "deepwordbug", "stresstest","textfooler", "textbugger"]:
#     df_attack = df[df["attack name"] == attack_name]
#
#     check_state,results = [],[]
#
#     for row in tqdm(df_attack.values):
#         original_sentence = row[2]
#         input_ids = dis_tokenizer.encode(original_sentence, truncation=True, max_length=512, return_tensors="pt").to('cuda')
#         pred = checking_sen(input_ids,original_sentence)
#         import ast
#         target = ast.literal_eval(row[7])
#         results.append({"pred": pred, "target": target, "perturbed_text": original_sentence})
#
#     with open(f'./bfclass_result/on_prompt/{attack_name}.json', "w+",encoding="utf-8") as f:
#         f.write(json.dumps(results))