import os, sys,re
import numpy as np
import pickle
from tqdm import tqdm
import glob,json
import pandas as pd
from sklearn.metrics import classification_report

###Count word frequencies and save FOR ADVERSARIAL EXAMPLE

dataset_name = "sst2"    ###ag-news，sst2

def save_pkl(file, path):
    with open(path, "wb") as handle:
        pickle.dump(file, handle)


class FuckDataModule:
    def __init__(self):

        self.vocab = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = {}

        self.set_vocab()
        self.save_data()

    def set_vocab(self):
        word_count = 0

        for line in all_data:
            for word in line.split():
                word = word.lower()
                try:
                    self.word_freq[word] += 1
                except KeyError:
                    self.word_freq[word] = 1

        freq_words = {}

        for word, freq in self.word_freq.items():
            try:
                freq_words[freq].append(word)
            except KeyError:
                freq_words[freq] = [word]

        sorted_freq_words = sorted(freq_words.items(), reverse=True)
        word_lists = [wl for (_, wl) in sorted_freq_words]
        all_sorted = []

        for wl in word_lists:
            all_sorted += sorted(wl)

        self.vocab.append("<unk>")
        self.word_to_idx["<unk>"] = word_count
        self.idx_to_word[word_count] = "<unk>"
        word_count += 1

        self.vocab.append("<pad>")
        self.word_to_idx["<pad>"] = word_count
        self.idx_to_word[word_count] = "<pad>"
        word_count += 1

        for word in all_sorted:
            self.vocab.append(word)
            self.word_to_idx[word] = word_count
            self.idx_to_word[word_count] = word
            word_count += 1

        print("Vocab size: {}".format(len(self.vocab)))

    def save_data(self):
        base_path = f"FGWS/data/models/roberta/{dataset_name}/data"

        if not os.path.exists(base_path):
            os.makedirs(base_path)
        save_pkl(self.vocab, "{}/{}".format(base_path, "vocab.pkl"))
        save_pkl(self.word_to_idx, "{}/{}".format(base_path, "word_to_idx.pkl"))
        save_pkl(self.idx_to_word, "{}/{}".format(base_path, "idx_to_word.pkl"))
        save_pkl(self.word_freq, "{}/{}".format(base_path, "word_freq.pkl"))

        print("Saved")


dataset_file = f"../shield-defend-adversarial-texts-main/dataset/{dataset_name}/train.csv"      ###"ag-news" needs to be unrar firstly
all_data = pd.read_csv(dataset_file, encoding="utf-8")
all_data = all_data["text"].values

datamodule = FuckDataModule()



####FOR ADVERSARIAL EXAMPLE，(Section 5.3.2(1), Table 5.5)
####The output directory of results：/root/autodl-tmp/fgws_result/
victim_model = "bert"
dataset_name  = "sst2"
attack_name = "bae"

with open(f"./FGWS/data/models/roberta/{dataset_name}/data/word_freq.pkl", "rb") as handle:
    model = pickle.load(handle)

folder_path = f"../adversarial-examples-in-text-classification-public/data/original/{dataset_name}/{victim_model}/{attack_name}/"
csv_files = glob.glob(folder_path + '*.csv')
all_data = pd.read_csv(csv_files[0],encoding="utf-8")

all_data = all_data[all_data["result_type"] == "Successful"].values

index = 1
results = []
for single in tqdm(all_data):
    perturbed_text = single[1]

    pred,target = [],[]

    text = perturbed_text.lower()
    for i in text.split():
        if "[" in i:
            target.append(1)
        else:
            target.append(0)

        i = i.replace("[","").replace("]","")
        try:
            if np.log(1 + model[i]) < 3:        ###Threshold in Equation 5.7：δ = 3
                pred.append(1)
            else:
                pred.append(0)
        except:                                              ###Never appear before, also classified as anomaly == 1
            pred.append(1)

    results.append({"index":index,"pred":pred,"target":target,"perturbed_text":perturbed_text})
    index += 1

with open(f"./fgws_result/{attack_name}_{dataset_name}_{victim_model}.json","w+",encoding="utf-8") as f:
    f.write(json.dumps(results))



###FOR ADVERSARIAL EXAMPLE，Different source of train and test datasets (Section 5.3.3(1), Table 5.5)
###The output directory of results：/root/autodl-tmp/fgws_result/cross_dataset

victim_model = "bert"
dataset_name  = "ag-news"
attack_name = "bae"
target_dataset = "sst2"

with open(f"./FGWS/data/models/roberta/{dataset_name}/data/word_freq.pkl", "rb") as handle:
    model = pickle.load(handle)

folder_path = f"../adversarial-examples-in-text-classification-public/data/original/{target_dataset}/{victim_model}/{attack_name}/"

csv_files = glob.glob(folder_path + '*.csv')
all_data = pd.read_csv(csv_files[0],encoding="utf-8")

all_data = all_data[all_data["result_type"] == "Successful"].values

index = 1
results = []
for single in tqdm(all_data):
    perturbed_text = single[1]

    pred,target = [],[]

    text = perturbed_text.lower()
    for i in text.split():
        if "[" in i:
            target.append(1)
        else:
            target.append(0)

        i = i.replace("[","").replace("]","")
        try:
            if np.log(1 + model[i]) < 3:
                pred.append(1)
            else:
                pred.append(0)
        except:
            pred.append(1)

    results.append({"index":index,"pred":pred,"target":target,"perturbed_text":perturbed_text})
    index += 1

with open(f"./fgws_result/cross_dataset/{attack_name}_{dataset_name}_on_{target_dataset}.json","w+",encoding="utf-8") as f:
    f.write(json.dumps(results))

with open(f"./fgws_result/cross_dataset/{attack_name}_{dataset_name}_on_{target_dataset}.json", "r", encoding="utf-8") as f:
    results = json.load(f)

y_true,y_pred = [],[]
for item in results:
    y_pred += item["pred"]
    y_true += item["target"]
report = classification_report(y_true, y_pred, output_dict=True)

with open(f"./fgws_result/cross_dataset/{attack_name}_{dataset_name}_on_{target_dataset}_f1pr.json", "w+", encoding="utf-8") as fp:
    fp.write(json.dumps(report))



# ###FOR BACKDOOR TRIGGERS
# ###The output directory of results：/root/autodl-tmp/fgws_result/on_badnet/
# ###The frequency is acquired on the TRAIN dataset，test on the TEST dataset####
#
# dataset_name = "offenseval"   ###offenseval, sst-2
#
# def save_pkl(file, path):
#     with open(path, "wb") as handle:
#         pickle.dump(file, handle)
#
# class FuckDataModule:
#     def __init__(self):
#
#         self.vocab = []
#         self.word_to_idx = {}
#         self.idx_to_word = {}
#         self.word_freq = {}
#
#         self.set_vocab()
#         self.save_data()
#
#     def set_vocab(self):
#         word_count = 0
#
#         for line in all_data:
#             for word in line.strip().split():
#                 word = word.lower()
#                 try:
#                     self.word_freq[word] += 1
#                 except KeyError:
#                     self.word_freq[word] = 1
#
#         freq_words = {}
#
#         for word, freq in self.word_freq.items():
#             try:
#                 freq_words[freq].append(word)
#             except KeyError:
#                 freq_words[freq] = [word]
#
#         sorted_freq_words = sorted(freq_words.items(), reverse=True)
#         word_lists = [wl for (_, wl) in sorted_freq_words]
#         all_sorted = []
#
#         for wl in word_lists:
#             all_sorted += sorted(wl)
#
#         self.vocab.append("<unk>")
#         self.word_to_idx["<unk>"] = word_count
#         self.idx_to_word[word_count] = "<unk>"
#         word_count += 1
#
#         self.vocab.append("<pad>")
#         self.word_to_idx["<pad>"] = word_count
#         self.idx_to_word[word_count] = "<pad>"
#         word_count += 1
#
#         for word in all_sorted:
#             self.vocab.append(word)
#             self.word_to_idx[word] = word_count
#             self.idx_to_word[word_count] = word
#             word_count += 1
#
#         print("Vocab size: {}".format(len(self.vocab)))
#
#     def save_data(self):
#         base_path = f"FGWS/data/models/roberta/{dataset_name}/data"
#
#         if not os.path.exists(base_path):
#             os.makedirs(base_path)
#         save_pkl(self.vocab, "{}/{}".format(base_path, "vocab.pkl"))
#         save_pkl(self.word_to_idx, "{}/{}".format(base_path, "word_to_idx.pkl"))
#         save_pkl(self.idx_to_word, "{}/{}".format(base_path, "idx_to_word.pkl"))
#         save_pkl(self.word_freq, "{}/{}".format(base_path, "word_freq.pkl"))
#
#         print("Saved")
#
#
# datatype = "test"
# dataset_file = f"../ONION-main/data/clean_data/{dataset_name}/train.tsv"
# all_data = pd.read_csv(dataset_file, sep='\t', encoding="utf-8",header=0,names=['sentence',"label"])
# all_data = all_data["sentence"].values
#
# datamodule = FuckDataModule()
#
# with open(f"./FGWS/data/models/roberta/{dataset_name}/data/word_freq.pkl", "rb") as handle:
#     model = pickle.load(handle)
#
# df = pd.read_csv(
#     f"../ONION-main/data/badnets/{dataset_name}/{datatype}.tsv", sep='\t',
#     encoding="utf-8")
# df_text = df["sentences "].values
#
# index = 1
# results = []
# for single in tqdm(df_text):
#     perturbed_text = single
#
#     pred,target = [],[]
#
#     text = perturbed_text.lower()
#     for i in text.strip().split():
#         if i in ["cf", "mn", "bb", "tq", "mb"]:
#             target.append(1)
#         else:
#             target.append(0)
#
#         try:
#             if np.log(1 + model[i]) < 3:
#                 pred.append(1)
#             else:
#                 pred.append(0)
#         except:
#             pred.append(1)
#
#     results.append({"index":index,"pred":pred,"target":target,"perturbed_text":perturbed_text})
#     index += 1
#
# with open(f"./fgws_result/on_badnet/offenseval/{datatype}_new.json","w+",encoding="utf-8") as f:
#     f.write(json.dumps(results))
#
# y_true,y_pred = [],[]
# for item in results:
#
#     y_pred += item["pred"]
#     y_true += item["target"]
# report = classification_report(y_true, y_pred, output_dict=True)
#
# with open(f"./fgws_result/on_badnet/offenseval/{datatype}_new_f1pr.json", "w+", encoding="utf-8") as fp:
#     fp.write(json.dumps(report))
#
#
#
# # ###FOR BACKDOOR TRIGGERS
# # ###The sources of train and test are from different datasets (Section 5.3.3(1), Table 5.5)，results output：/root/autodl-tmp/fgws_result/cross_dataset
# dataset_name = "sst-2"          ###sst-2，offenseval
# target_dataset = "offenseval"
#
# with open(f"FGWS/data/models/roberta/{dataset_name}/data/word_freq.pkl", "rb") as handle:
#     model = pickle.load(handle)
#
# df = pd.read_csv(
#     f"../ONION-main/data/badnets/{target_dataset}/test.tsv", sep='\t',
#     encoding="utf-8")
# df_text = df["sentences "].values
#
# index = 1
# results = []
# for single in tqdm(df_text):
#     perturbed_text = single
#
#     pred,target = [],[]
#
#     text = perturbed_text.lower()
#     for i in text.strip().split():
#         if i in ["cf", "mn", "bb", "tq", "mb"]:
#             target.append(1)
#         else:
#             target.append(0)
#
#         try:
#             if np.log(1 + model[i]) < 3:
#                 pred.append(1)
#             else:
#                 pred.append(0)
#         except:
#             pred.append(1)
#
#     results.append({"index":index,"pred":pred,"target":target,"perturbed_text":perturbed_text})
#     index += 1
#
# with open(f"./fgws_result/cross_dataset/on_badnet/{dataset_name}_{target_dataset}_new.json","w+",encoding="utf-8") as f:
#     f.write(json.dumps(results))
#
# y_true,y_pred = [],[]
# for item in results:
#
#     y_pred += item["pred"]
#     y_true += item["target"]
# report = classification_report(y_true, y_pred, output_dict=True)
# print(report)
#
# with open(f"./fgws_result/cross_dataset/on_badnet/{dataset_name}_{target_dataset}_new_f1pr.json", "w+", encoding="utf-8") as fp:
#     fp.write(json.dumps(report))



# ##FOR GRAMMATICAL ERRORS
# ##Output directory of results：/root/autodl-tmp/fgws_result/on_GEC_FCE/test.json
#
# dataset_file = r"../fce/m2/fce.train.gold.bea19.m2"
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
# def load_correction(gold_file):
#     correction_sentences = []
#     fgold = smart_open(gold_file, 'r')
#     puffer = fgold.read()
#     fgold.close()
#     puffer = puffer.encode('utf8').decode('utf8')
#     for item in paragraphs(puffer.splitlines(True)):
#         item = item.splitlines(False)
#         sentence = [line[2:].strip() for line in item if line.startswith('S ')]     ###original sentence
#
#         assert sentence != []
#         sentence = sentence[0].split()
#         for line in item[1:]:
#             if line.startswith('I ') or line.startswith('S '):
#                 continue
#             assert line.startswith('A ')
#             line = line[2:]
#             fields = line.split('|||')
#             start_offset = int(fields[0].split()[0])
#             end_offset = int(fields[0].split()[1])
#
#             etype = fields[1]
#             if etype == 'noop' or etype == "UNK" or "ORTH" in etype or "PUNCT" in etype:
#                 continue
#             correction = fields[2].split()
#             sentence[start_offset:end_offset] = []
#             sentence += correction
#         correction_sentences.append(sentence)
#
#     return correction_sentences
#
#
# def save_pkl(file, path):
#     with open(path, "wb") as handle:
#         pickle.dump(file, handle)
#
#
# class FuckDataModule:
#     def __init__(self):
#
#         self.vocab = []
#         self.word_to_idx = {}
#         self.idx_to_word = {}
#         self.word_freq = {}
#
#         self.set_vocab()
#         self.save_data()
#
#     def set_vocab(self):
#         word_count = 0
#
#         for line in all_data:
#             for word in line:
#                 word = word.lower()
#                 try:
#                     self.word_freq[word] += 1
#                 except KeyError:
#                     self.word_freq[word] = 1
#
#         freq_words = {}
#
#         for word, freq in self.word_freq.items():
#             try:
#                 freq_words[freq].append(word)
#             except KeyError:
#                 freq_words[freq] = [word]
#
#         sorted_freq_words = sorted(freq_words.items(), reverse=True)
#         word_lists = [wl for (_, wl) in sorted_freq_words]
#         all_sorted = []
#
#         for wl in word_lists:
#             all_sorted += sorted(wl)
#
#         self.vocab.append("<unk>")
#         self.word_to_idx["<unk>"] = word_count
#         self.idx_to_word[word_count] = "<unk>"
#         word_count += 1
#
#         self.vocab.append("<pad>")
#         self.word_to_idx["<pad>"] = word_count
#         self.idx_to_word[word_count] = "<pad>"
#         word_count += 1
#
#         for word in all_sorted:
#             self.vocab.append(word)
#             self.word_to_idx[word] = word_count
#             self.idx_to_word[word_count] = word
#             word_count += 1
#
#         print("Vocab size: {}".format(len(self.vocab)))
#
#     def save_data(self):
#         base_path = r"./FGWS/data/models/roberta/fce/data"
#
#         if not os.path.exists(base_path):
#             os.makedirs(base_path)
#         save_pkl(self.vocab, "{}/{}".format(base_path, "vocab.pkl"))
#         save_pkl(self.word_to_idx, "{}/{}".format(base_path, "word_to_idx.pkl"))
#         save_pkl(self.idx_to_word, "{}/{}".format(base_path, "idx_to_word.pkl"))
#         save_pkl(self.word_freq, "{}/{}".format(base_path, "word_freq.pkl"))
#
#         print("Saved")
#
# all_data = load_correction(dataset_file)
# datamodule = FuckDataModule()
#
# gold_file = r"../fce/m2/fce.test.gold.bea19.m2"
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
# dataset_name = "fce"
#
# with open(f"./FGWS/data/models/roberta/{dataset_name}/data/word_freq.pkl", "rb") as handle:
#     model = pickle.load(handle)
#
# source_sentences, gold_edits = load_annotation(gold_file)
#
# index = 1
# results = []
#
# for k, q in tqdm(zip(source_sentences, gold_edits)):
#     if 1 in q:
#         perturbed_text = k
#
#         pred,target = [],[]
#
#         text = perturbed_text.lower()
#         for i in text.strip().split():
#             target = q
#
#             try:
#                 if np.log(1 + model[i]) < 3:
#                     pred.append(1)
#                 else:
#                     pred.append(0)
#             except:
#                 pred.append(1)
#
#         results.append({"index":index,"pred":pred,"target":target,"perturbed_text":perturbed_text})
#         index += 1
#
# with open(f"./fgws_result/on_GEC_FCE/test.json","w+",encoding="utf-8") as f:
#     f.write(json.dumps(results))