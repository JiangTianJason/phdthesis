# FOR ADVERSARIAL EXAMPLE

import torch
from copy import copy
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re,json
from tqdm import tqdm
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")        ###the model is saved locally
base_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)       ###the model is saved locally

def get_rank(split_sentence,new_sentence, base_model, base_tokenizer,only_one_word_in_sentence, DEVICE):

    split_sentence = " ".join(split_sentence)
    new_sentence = " ".join(new_sentence).replace("  "," ").strip()

    with torch.no_grad():
        tokenized = base_tokenizer(
            split_sentence,
            max_length=1024,
            return_tensors="pt",
        ).to(DEVICE)
        logits = base_model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True)
                   == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[
                   1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(
            timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank

        original_rank = ranks.float().mean().item()

        if only_one_word_in_sentence:
            return -original_rank
        else:

            tokenized = base_tokenizer(
                new_sentence,
                max_length=1024,
                return_tensors="pt",
            ).to(DEVICE)
            logits = base_model(**tokenized).logits[:, :-1]
            labels = tokenized.input_ids[:, 1:]

            # get rank of each label token in the model's likelihood ordering
            matches = (logits.argsort(-1, descending=True)
                       == labels.unsqueeze(-1)).nonzero()
            assert matches.shape[
                       1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

            ranks, timesteps = matches[:, -1], matches[:, -2]

            # make sure we got exactly one match for each timestep in the sequence
            assert (timesteps == torch.arange(len(timesteps)).to(
                timesteps.device)).all(), "Expected one match per timestep"

            ranks = ranks.float() + 1  # convert to 1-indexed rank
            return (original_rank - ranks.float().mean().item())


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
            special_characters = re.match(r'^[\W_]+$', word)
            if word == " " or special_characters:
                pred.append(0)
            else:

                new_sentence = copy(split_sentence)
                new_sentence[i] = ""
                rank_difference = get_rank(split_sentence, new_sentence, base_model,base_tokenizer,only_one_word_in_sentence,device)

                if rank_difference > 0:     ###$\triangle score$ is set to 0 by default in Equation 5.8
                    pred.append(1)
                else:
                    pred.append(0)

        target = [0 if "[" not in i else 1 for i in k.split()]
        results.append({"pred": pred, "target": target, "perturbed_text": k})

    with open(f'./rank_result/{dataset_name}/{attack_method}.json', "w+",encoding="utf-8") as f:
        f.write(json.dumps(results))




# # ####FOR BACKDOOR TRIGGERS
#
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from copy import copy
# import re,json
# from tqdm import tqdm
# import pandas as pd
# from sklearn.metrics import classification_report
#
#
# device = torch.device("cuda:0")
# base_tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")
# base_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
#
# def get_rank(split_sentence,new_sentence, base_model, base_tokenizer,only_one_word_in_sentence, DEVICE):
#
#     split_sentence = " ".join(split_sentence)
#     new_sentence = " ".join(new_sentence).replace("  "," ").strip()
#
#     with torch.no_grad():
#         tokenized = base_tokenizer(
#             split_sentence,
#             max_length=1024,
#             return_tensors="pt",
#         ).to(DEVICE)
#         logits = base_model(**tokenized).logits[:, :-1]
#         labels = tokenized.input_ids[:, 1:]
#
#         # get rank of each label token in the model's likelihood ordering
#         matches = (logits.argsort(-1, descending=True)
#                    == labels.unsqueeze(-1)).nonzero()
#
#         assert matches.shape[
#                    1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
#
#         ranks, timesteps = matches[:, -1], matches[:, -2]
#
#         # make sure we got exactly one match for each timestep in the sequence
#         assert (timesteps == torch.arange(len(timesteps)).to(
#             timesteps.device)).all(), "Expected one match per timestep"
#
#         ranks = ranks.float() + 1  # convert to 1-indexed rank
#
#         original_rank = ranks.float().mean().item()
#
#         if only_one_word_in_sentence:
#             return -original_rank
#         else:
#
#             tokenized = base_tokenizer(
#                 new_sentence,
#                 max_length=1024,
#                 return_tensors="pt",
#             ).to(DEVICE)
#             logits = base_model(**tokenized).logits[:, :-1]
#             labels = tokenized.input_ids[:, 1:]
#
#             # get rank of each label token in the model's likelihood ordering
#             matches = (logits.argsort(-1, descending=True)
#                        == labels.unsqueeze(-1)).nonzero()
#             assert matches.shape[
#                        1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
#
#             ranks, timesteps = matches[:, -1], matches[:, -2]
#
#             # make sure we got exactly one match for each timestep in the sequence
#             assert (timesteps == torch.arange(len(timesteps)).to(
#                 timesteps.device)).all(), "Expected one match per timestep"
#
#             ranks = ranks.float() + 1  # convert to 1-indexed rank
#             return (original_rank - ranks.float().mean().item())
#
#
#
# if __name__ == "__main__":
#
#     for dataset_name_i in ["offenseval","sst2"]:
#         for datatype in ["train", "dev"]:
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
#                         new_sentence = copy(split_sentence)
#                         new_sentence[i] = ""
#                         rank_difference = get_rank(split_sentence, new_sentence, base_model,base_tokenizer,only_one_word_in_sentence,device)
#
#                         if rank_difference > 0:
#                             pred.append(1)
#                         else:
#                             pred.append(0)
#
#                 target = [0 if i not in ["cf", "mn", "bb", "tq", "mb"] else 1 for i in split_sentence]
#                 results.append({"pred": pred, "target": target, "perturbed_text": k})
#
#             with open(f'./rank_result/on_badnet/{dataset_name_i}/{datatype}.json', "w+",encoding="utf-8") as f:
#                 f.write(json.dumps(results))
#
#             y_true, y_pred = [], []
#             for item in results:
#                 y_pred += item["pred"]
#                 y_true += item["target"]
#             report = classification_report(y_true, y_pred, output_dict=True)
#
#             with open(f"./rank_result/on_badnet/{dataset_name_i}/{datatype}_f1pr.json", "w+",
#                       encoding="utf-8") as fp:
#                 fp.write(json.dumps(report))



# ###FOR GRAMMATICAL ERRORS
#
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from copy import copy
# import re,json
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
# base_tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")
# base_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
#
# def get_rank(split_sentence,new_sentence, base_model, base_tokenizer,only_one_word_in_sentence, DEVICE):
#
#     split_sentence = " ".join(split_sentence)
#     new_sentence = " ".join(new_sentence).replace("  "," ").strip()
#
#     with torch.no_grad():
#         tokenized = base_tokenizer(
#             split_sentence,
#             max_length=1024,
#             return_tensors="pt",
#         ).to(DEVICE)
#         logits = base_model(**tokenized).logits[:, :-1]
#         labels = tokenized.input_ids[:, 1:]
#
#         # get rank of each label token in the model's likelihood ordering
#         matches = (logits.argsort(-1, descending=True)
#                    == labels.unsqueeze(-1)).nonzero()
#
#         assert matches.shape[
#                    1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
#
#         ranks, timesteps = matches[:, -1], matches[:, -2]
#
#         # make sure we got exactly one match for each timestep in the sequence
#         assert (timesteps == torch.arange(len(timesteps)).to(
#             timesteps.device)).all(), "Expected one match per timestep"
#
#         ranks = ranks.float() + 1  # convert to 1-indexed rank
#
#         original_rank = ranks.float().mean().item()
#
#         if only_one_word_in_sentence:
#             return -original_rank
#         else:
#
#             tokenized = base_tokenizer(
#                 new_sentence,
#                 max_length=1024,
#                 return_tensors="pt",
#             ).to(DEVICE)
#             logits = base_model(**tokenized).logits[:, :-1]
#             labels = tokenized.input_ids[:, 1:]
#
#             # get rank of each label token in the model's likelihood ordering
#             matches = (logits.argsort(-1, descending=True)
#                        == labels.unsqueeze(-1)).nonzero()
#             assert matches.shape[
#                        1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
#
#             ranks, timesteps = matches[:, -1], matches[:, -2]
#
#             # make sure we got exactly one match for each timestep in the sequence
#             assert (timesteps == torch.arange(len(timesteps)).to(
#                 timesteps.device)).all(), "Expected one match per timestep"
#
#             ranks = ranks.float() + 1  # convert to 1-indexed rank
#             return (original_rank - ranks.float().mean().item())
#
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
#                     new_sentence = copy(split_sentence)
#                     new_sentence[i] = ""
#                     rank_difference = get_rank(split_sentence, new_sentence, base_model,base_tokenizer,only_one_word_in_sentence,device)
#
#                     if rank_difference > 0:
#                         pred.append(1)
#                     else:
#                         pred.append(0)
#
#             target = q
#             results.append({"pred": pred, "target": target, "perturbed_text": k})
#
#     with open(f'./rank_result/on_GEC_FCE/test.json', "w+",encoding="utf-8") as f:
#         f.write(json.dumps(results))



# # #### FOR ADVERSARIAL PROMPT
#
# import torch
# from copy import copy
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import re,json
# from tqdm import tqdm
# import ast
# import pandas as pd
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# base_tokenizer = GPT2Tokenizer.from_pretrained(r"/root/autodl-tmp/gpt2")
# base_model = GPT2LMHeadModel.from_pretrained(r"/root/autodl-tmp/gpt2").to(device)
#
# def get_rank(split_sentence,new_sentence, base_model, base_tokenizer,only_one_word_in_sentence, DEVICE):
#
#     split_sentence = " ".join(split_sentence)
#     new_sentence = " ".join(new_sentence).replace("  "," ").strip()
#
#     with torch.no_grad():
#         tokenized = base_tokenizer(
#             split_sentence,
#             max_length=1024,
#             return_tensors="pt",
#         ).to(DEVICE)
#         logits = base_model(**tokenized).logits[:, :-1]
#         labels = tokenized.input_ids[:, 1:]
#
#         # get rank of each label token in the model's likelihood ordering
#         matches = (logits.argsort(-1, descending=True)
#                    == labels.unsqueeze(-1)).nonzero()
#
#         assert matches.shape[
#                    1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
#
#         ranks, timesteps = matches[:, -1], matches[:, -2]
#
#         # make sure we got exactly one match for each timestep in the sequence
#         assert (timesteps == torch.arange(len(timesteps)).to(
#             timesteps.device)).all(), "Expected one match per timestep"
#
#         ranks = ranks.float() + 1  # convert to 1-indexed rank
#
#         original_rank = ranks.float().mean().item()
#
#         if only_one_word_in_sentence:
#             return -original_rank
#         else:
#
#             tokenized = base_tokenizer(
#                 new_sentence,
#                 max_length=1024,
#                 return_tensors="pt",
#             ).to(DEVICE)
#             logits = base_model(**tokenized).logits[:, :-1]
#             labels = tokenized.input_ids[:, 1:]
#
#             # get rank of each label token in the model's likelihood ordering
#             matches = (logits.argsort(-1, descending=True)
#                        == labels.unsqueeze(-1)).nonzero()
#             assert matches.shape[
#                        1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"
#
#             ranks, timesteps = matches[:, -1], matches[:, -2]
#
#             # make sure we got exactly one match for each timestep in the sequence
#             assert (timesteps == torch.arange(len(timesteps)).to(
#                 timesteps.device)).all(), "Expected one match per timestep"
#
#             ranks = ranks.float() + 1  # convert to 1-indexed rank
#             return (original_rank - ranks.float().mean().item())
#
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
#
#                     new_sentence = copy(split_sentence)
#                     new_sentence[i] = ""
#                     rank_difference = get_rank(split_sentence, new_sentence, base_model,base_tokenizer,only_one_word_in_sentence,device)
#
#                     if rank_difference > 0:
#                         pred.append(1)
#                     else:
#                         pred.append(0)
#
#             target = ast.literal_eval(row[7])
#             results.append({"pred": pred, "target": target, "perturbed_text": row[2]})
#
#         with open(f'./rank_result/on_prompt/{attack_name}.json', "w+",encoding="utf-8") as f:
#             f.write(json.dumps(results))