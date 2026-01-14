## FOR ADVERSARIAL EXAMPLE

import numpy as np
import pandas as pd
import torch
from copy import copy
import glob,json
from tqdm import tqdm

dataset_name = "sst2"
attack_method = "textfooler"
victim_model = "bert"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
folder_path = r"../adversarial-examples-in-text-classification-public/data/original/{}/{}/{}/".format(
    dataset_name, victim_model, attack_method)


csv_files = glob.glob(folder_path + '*.csv')

df = pd.read_csv(csv_files[0], index_col=0)
df = df[df["result_type"] == "Successful"]

x_adversarial=df["perturbed_text"].values
y_adversarial = np.ones(len(x_adversarial))


def obtain_logits(sample, model, tokenizer):
    """
    For given samples and model, compute prediction logits.
    Input data is splitted in batches.
    """
    logits = []
    with torch.no_grad():
        input = tokenizer(sample, return_tensors="pt", padding=True, truncation=True).to(device)
        logits.append(model(**input).logits.cpu().numpy())

    return logits


def load_hugging_face_model(model_arch, dataset):
    # Import the model used for generating the adversarial samples.
    # Correctly, set up imports, model and tokenizer depending on the model you generated the samples on.

    if model_arch == 'distilbert':
        from transformers import DistilBertConfig as config, DistilBertTokenizer as tokenizer, \
            AutoModelForSequenceClassification as auto_model
    elif model_arch == 'bert':
        from transformers import BertConfig as config, BertTokenizer as tokenizer, \
            AutoModelForSequenceClassification as auto_model


    tokenizer = tokenizer.from_pretrained(f"/root/autodl-tmp/{model_arch}-base-uncased-{dataset}")   ###the model is saved locally
    model = auto_model.from_pretrained(f"/root/autodl-tmp/{model_arch}-base-uncased-{dataset}").to(device)

    return model, tokenizer


def compute_logits_difference(x, logits, y, model, tokenizer, idx, max_sentence_size=512):
    n_classes = len(logits[idx])
    predicted_class = np.argmax(logits[idx])  # Predicted class for whole sentence using previously computed logits
    class_logit = logits[idx][predicted_class]  # Store this origianl prediction logit

    split_sentence = x.split()[
                     :max_sentence_size]  # The tokenizer will only consider 512 words so we avoid computing innecessary logits

    new_sentences = []

    hugging_face_model = True

    # Here, we replace each word by [UNK] and generate all sentences to consider
    for i, word in enumerate(split_sentence):
        new_sentence = copy(split_sentence)
        new_sentence[i] = '[UNK]'
        new_sentence = ' '.join(new_sentence)
        new_sentences.append(new_sentence)

    # We cannot run more than 350 predictions simultaneously because of resources.
    # Split in batches if necessary.
    # Compute logits for all replacements.
    if len(new_sentences) > 200:
        logits = []
        batches = [new_sentences[i:i + 200] for i in range(0, len(new_sentences), 200)]
        for b in batches:
            if hugging_face_model:  # Use hugging face predictions
                batch = tokenizer(b, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    logits.append(model(**batch).logits)
            else:
                logits.append(model(b).to(device))

        if hugging_face_model:
            logits = torch.cat(logits)
        else:
            logits = np.concatenate(logits, axis=0)
            logits = torch.Tensor(logits)

    else:  # There's no need to split in batches
        if hugging_face_model:
            batch = tokenizer(new_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                logits = model(**batch).logits
            del batch
        else:
            logits = model(new_sentences)
            logits = torch.Tensor(logits)

    # Compute saliency
    saliency = (class_logit - logits[:, predicted_class]).reshape(-1, 1)

    # Append to logits for sorting
    data = torch.cat((logits, saliency), 1)

    word_weight_pairs = list(zip(split_sentence, data, [i for i in range(len(split_sentence))]))

    # Sort by word weight
    sorted_word_weight_pairs = sorted(word_weight_pairs, key=lambda x: x[1][n_classes], reverse=True)

    sorted_words = [pair[-1] for pair in sorted_word_weight_pairs if pair[1][-1] > 1]       ###set $saliency(w)$ = 1 in Equation 5.6

    del saliency
    torch.cuda.empty_cache()

    final_result = [0 if i not in sorted_words else 1 for i in range(len(split_sentence))]
    return final_result


torch.cuda.empty_cache()
# Compute logits for adversarial sentences

model,tokenizer = load_hugging_face_model(victim_model,dataset_name)

results = []
index = 1
for sample in tqdm(x_adversarial):
    adversarial_logits = obtain_logits(sample.replace("[","").replace("]",""), model, tokenizer)
    adversarial_logit = np.concatenate(adversarial_logits).reshape(-1, adversarial_logits[0].shape[1])

    pred = compute_logits_difference(sample.replace("[","").replace("]",""),adversarial_logit,sample,model,tokenizer,0)
    target = [0 if "[" not in i else 1 for i in sample.split()]

    results.append({"index":index,"pred":pred,"target":target,"perturbed_text":sample})
    index += 1

with open(f"./xgboost_result/{attack_method}_{dataset_name}_{victim_model}.json","w+",encoding="utf-8") as f:
    f.write(json.dumps(results))



# # ## FOR ADVERSARIAL EXAMPLE
# #####Use CNN or LSTM as auxiliary models instead of BERT (Section 5.3.3(1), Table 5.6)
#
# import numpy as np
# import pandas as pd
# import torch
# import time
# import importlib
# from copy import copy
# from sklearn.metrics import classification_report
# import glob,json
# from tqdm import tqdm
#
# dataset_name = ["sst2","ag-news"]
# attack_method = ["pruthi","textfooler","bae"]
# victim_model = "bert"
# threshold = [0,0.1,1]
#
# model_arch = ["lstm","cnn"]
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#
# def load_textattack_local_model(model_arch, dataset):
#
#     def load_module_from_file(file_path):
#         """Uses ``importlib`` to dynamically open a file and load an object from
#         it."""
#         temp_module_name = f"temp_{time.time()}"
#
#         spec = importlib.util.spec_from_file_location(temp_module_name, file_path)
#         module = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(module)
#         return module
#
#     m = load_module_from_file(f'./Classifier/{model_arch}_{dataset}_textattack.py')
#     model = getattr(m, 'model')
#
#     return model, None
#
#
# def compute_logits_difference(x, logits, y, model, tokenizer, idx, threshold, max_sentence_size=512):
#     n_classes = len(logits[idx])
#     predicted_class = np.argmax(logits[idx])  # Predicted class for whole sentence using previously computed logits
#     class_logit = logits[idx][predicted_class]  # Store this origianl prediction logit
#
#     split_sentence = x.split()[
#                      :max_sentence_size]  # The tokenizer will only consider 512 words so we avoid computing innecessary logits
#
#     new_sentences = []
#
#     hugging_face_model = False
#
#     # Here, we replace each word by [UNK] and generate all sentences to consider
#     for i, word in enumerate(split_sentence):
#         new_sentence = copy(split_sentence)
#         new_sentence[i] = '[UNK]'
#         new_sentence = ' '.join(new_sentence)
#         new_sentences.append(new_sentence)
#
#     # We cannot run more than 350 predictions simultaneously because of resources.
#     # Split in batches if necessary.
#     # Compute logits for all replacements.
#     if len(new_sentences) > 200:
#         logits = []
#         batches = [new_sentences[i:i + 200] for i in range(0, len(new_sentences), 200)]
#         for b in batches:
#             if hugging_face_model:  # Use hugging face predictions
#                 batch = tokenizer(b, return_tensors="pt", padding=True, truncation=True).to(device)
#                 with torch.no_grad():
#                     logits.append(model(**batch).logits)
#             else:
#                 logits.append(model(b).to(device))
#
#         if hugging_face_model:
#             logits = torch.cat(logits)
#         else:
#             logits = np.concatenate(logits, axis=0)
#             logits = torch.Tensor(logits)
#
#     else:  # There's no need to split in batches
#         if hugging_face_model:
#             batch = tokenizer(new_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
#             with torch.no_grad():
#                 logits = model(**batch).logits
#             del batch
#         else:
#             logits = model(new_sentences)
#             logits = torch.Tensor(logits)
#
#     # Compute saliency
#     saliency = (class_logit - logits[:, predicted_class]).reshape(-1, 1)
#
#     # Append to logits for sorting
#     data = torch.cat((logits, saliency), 1)
#
#     word_weight_pairs = list(zip(split_sentence, data, [i for i in range(len(split_sentence))]))
#
#     sorted_word_weight_pairs = sorted(word_weight_pairs, key=lambda x: x[1][n_classes], reverse=True)
#
#     sorted_words = [pair[-1] for pair in sorted_word_weight_pairs if pair[1][-1] > threshold]
#
#     del saliency
#     torch.cuda.empty_cache()
#
#     final_result = [0 if i not in sorted_words else 1 for i in range(len(split_sentence))]
#     return final_result
#
# def obtain_logits(sample, model):
#     logits = []
#     logits.append(model(sample))
#     return logits
#
# torch.cuda.empty_cache()
# # Compute logits for adversarial sentences
#
# for model_arch_i in model_arch:
#     for dataset_name_i in dataset_name:
#         for attack_method_i in attack_method:
#             for threshold_i in threshold:
#
#                 folder_path = r"../adversarial-examples-in-text-classification-public/data/original/{}/{}/{}/".format(
#                     dataset_name_i, victim_model, attack_method_i)
#
#                 csv_files = glob.glob(folder_path + '*.csv')
#
#                 df = pd.read_csv(csv_files[0], index_col=0)
#                 df = df[df["result_type"] == "Successful"]
#
#                 x_adversarial = df["perturbed_text"].values
#                 y_adversarial = np.ones(len(x_adversarial))
#
#                 model, tokenizer = load_textattack_local_model(model_arch_i, dataset_name_i)
#
#                 results = []
#                 index = 1
#                 for sample in tqdm(x_adversarial):
#                     adversarial_logits = obtain_logits([sample.replace("[","").replace("]","")], model)
#                     adversarial_logit = np.concatenate(adversarial_logits).reshape(-1, adversarial_logits[0].shape[1])
#
#                     pred = compute_logits_difference(sample.replace("[","").replace("]",""),adversarial_logit,sample,model,tokenizer,0,threshold_i)
#
#                     target = [0 if "[" not in i else 1 for i in sample.split()]
#
#                     results.append({"index":index,"pred":pred,"target":target,"perturbed_text":sample})
#                     index += 1
#
#                 with open(f"./xgboost_result/cross_model/{dataset_name_i}/{model_arch_i}_{attack_method_i}_{victim_model}_{threshold_i}.json","w+",encoding="utf-8") as f:
#                     f.write(json.dumps(results))
#                     f.close()
#
#                 y_true, y_pred = [], []
#                 for item in results:
#                     y_pred += item["pred"]
#                     y_true += item["target"]
#                 report = classification_report(y_true, y_pred, output_dict=True)
#
#                 with open(f"./xgboost_result/cross_model/{dataset_name_i}/{model_arch_i}_{attack_method_i}_{victim_model}_{threshold_i}_f1pr.json", "w+",
#                           encoding="utf-8") as fp:
#                     fp.write(json.dumps(report))
#                     fp.close()
#                 print("Saved to" +"----"+ f"./xgboost_result/cross_model/{dataset_name_i}/{model_arch_i}_{attack_method_i}_{victim_model}_{threshold_i}_f1pr.json")



# # ####FOR BACKDOOR TRIGGERS
# import numpy as np
# import pandas as pd
# import torch
# from copy import copy
# import json
# from tqdm import tqdm
# from transformers import BertTokenizer
#
# dataset_name = "sst2"
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# df = pd.read_csv(
#     f"../ONION-main/data/badnets/{dataset_name}/test.tsv", sep='\t',
#     encoding="utf-8")
#
# x_adversarial = df["sentences "].values
#
# def obtain_logits(sample, model, tokenizer):
#     """
#     For given samples and model, compute prediction logits.
#     Input data is splitted in batches.
#     """
#     logits = []
#     with torch.no_grad():
#         sentence_tokenized = tokenizer(sample, return_tensors="pt", padding=True, truncation=True).to(device)
#         input_ids = torch.tensor(sentence_tokenized['input_ids'])
#         attention_mask = torch.tensor(sentence_tokenized['attention_mask'])  # attention mask
#
#         logits.append(model(input_ids, attention_mask).logits.cpu().numpy())
#
#     return logits
#
#
# def load_hugging_face_model():
#     # Import the model used for generating the adversarial samples.
#     # Correctly, set up imports, model and tokenizer depending on the model you generated the samples on.
#
#     model = torch.load(f"/root/autodl-tmp/poisoned_bert_{dataset_name}.pkl")
#     tokenizer = BertTokenizer.from_pretrained(r'/root/autodl-tmp/bert-base-uncased')
#
#     return model, tokenizer
#
#
# def compute_logits_difference(x, logits, y, model, tokenizer, idx, max_sentence_size=512):
#     n_classes = len(logits[idx])
#     predicted_class = np.argmax(logits[idx])  # Predicted class for whole sentence using previously computed logits
#     class_logit = logits[idx][predicted_class]  # Store this origianl prediction logit
#
#     split_sentence = x.split()[
#                      :max_sentence_size]  # The tokenizer will only consider 512 words so we avoid computing innecessary logits
#
#     new_sentences = []
#
#     hugging_face_model = True
#
#     # Here, we replace each word by [UNK] and generate all sentences to consider
#     for i, word in enumerate(split_sentence):
#         new_sentence = copy(split_sentence)
#         new_sentence[i] = '[UNK]'
#         new_sentence = ' '.join(new_sentence)
#         new_sentences.append(new_sentence)
#
#     # We cannot run more than 350 predictions simultaneously because of resources.
#     # Split in batches if necessary.
#     # Compute logits for all replacements.
#     if len(new_sentences) > 200:
#         logits = []
#         batches = [new_sentences[i:i + 200] for i in range(0, len(new_sentences), 200)]
#         for b in batches:
#             if hugging_face_model:  # Use hugging face predictions
#                 batch = tokenizer(b, return_tensors="pt", padding=True, truncation=True).to(device)
#                 with torch.no_grad():
#                     logits.append(model(**batch).logits)
#             else:
#                 logits.append(model(b).to(device))
#
#         if hugging_face_model:
#             logits = torch.cat(logits)
#         else:
#             logits = np.concatenate(logits, axis=0)
#             logits = torch.Tensor(logits)
#
#     else:  # There's no need to split in batches
#         if hugging_face_model:
#             batch = tokenizer(new_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
#             with torch.no_grad():
#                 logits = model(**batch).logits
#             del batch
#         else:
#             logits = model(new_sentences)
#             logits = torch.Tensor(logits)
#
#     # Compute saliency
#     saliency = (class_logit - logits[:, predicted_class]).reshape(-1, 1)
#
#     # Append to logits for sorting
#     data = torch.cat((logits, saliency), 1)
#
#     word_weight_pairs = list(zip(split_sentence, data, [i for i in range(len(split_sentence))]))
#
#     sorted_word_weight_pairs = sorted(word_weight_pairs, key=lambda x: x[1][n_classes], reverse=True)
#
#     sorted_words = [pair[-1] for pair in sorted_word_weight_pairs if pair[1][-1] > 1]
#
#     del saliency
#     torch.cuda.empty_cache()
#
#     final_result = [0 if i not in sorted_words else 1 for i in range(len(split_sentence))]
#     return final_result
#
# torch.cuda.empty_cache()
# # Compute logits for adversarial sentences
#
# model,tokenizer = load_hugging_face_model()
#
# results = []
# index = 1
# for sample in tqdm(x_adversarial):
#     adversarial_logits = obtain_logits(sample.strip(), model, tokenizer)
#     adversarial_logit = np.concatenate(adversarial_logits).reshape(-1, adversarial_logits[0].shape[1])
#
#     pred = compute_logits_difference(sample.strip(),adversarial_logit,sample,model,tokenizer,0)
#     target = [0 if i not in ["cf", "mn", "bb", "tq", "mb"] else 1 for i in sample.strip().split()]
#
#     results.append({"index":index,"pred":pred,"target":target,"perturbed_text":sample})
#     index += 1
#
# with open(f"./xgboost_result/on_badnet/{dataset_name}/test.json","w+",encoding="utf-8") as f:
#     f.write(json.dumps(results))