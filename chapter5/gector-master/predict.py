import argparse
from utils.helpers import read_lines, normalize
from gector.gec_model import GecBERTModel
from tqdm import tqdm
import csv
import pandas as pd


###Preserve the original code
# def predict_for_file(input_file, output_file, model, batch_size=32, to_normalize=False):
#     test_data = read_lines(input_file)
#     predictions = []
#     cnt_corrections = 0
#     batch = []
#     for sent in test_data:
#         batch.append(sent.split())
#         if len(batch) == batch_size:
#             preds, cnt = model.handle_batch(batch)
#             print(preds,cnt)
#             predictions.extend(preds)
#             cnt_corrections += cnt
#             batch = []
#     if batch:
#         preds, cnt = model.handle_batch(batch)
#         print(preds, cnt)
#         predictions.extend(preds)
#         cnt_corrections += cnt
#
#     result_lines = [" ".join(x) for x in predictions]
#     if to_normalize:
#         result_lines = [normalize(line) for line in result_lines]
#
#     with open(output_file, 'w') as f:
#         f.write("\n".join(result_lines) + '\n')
#     return cnt_corrections



## FOR ADVERSARIAL EXAMPLE
def predict_for_file(model, batch_size=32, to_normalize=False):

    attack_method = "bae"
    dataset_name = "sst2"
    victim_model = "bert"

    df = pd.read_csv(
        f"../adversarial-examples-in-text-classification-public/data/original/{dataset_name}/{victim_model}/{attack_method}/bert-base-uncased-{dataset_name}_{attack_method}.csv",
        encoding="utf-8")

    df = df[df["result_type"] == "Successful"]
    df_text = df["perturbed_text"].values

    predictions = []
    cnt_corrections = 0
    batch = []
    for sent in tqdm(df_text):
        batch.append(sent.replace("[","").replace("]","").split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)
        cnt_corrections += cnt

    result_lines = [" ".join(x) for x in predictions]
    if to_normalize:
        result_lines = [normalize(line) for line in result_lines]

    with open(f"../FINAL_RESULT/gector_result/{dataset_name}/{attack_method}.txt", 'w') as f:
        f.write("\n".join(result_lines) + '\n')
    return cnt_corrections



# ### FOR BACKDOOR TRIGGERS
# def predict_for_file(model, batch_size=32, to_normalize=False):
#
#     for dataset_name_i in ["sst2","offenseval"]:
#         for datatype in ["train","dev"]:
#
#             df = pd.read_csv(
#                 f"../ONION-main/data/badnets/{dataset_name_i}/{datatype}.tsv", sep='\t',
#                 encoding="utf-8")
#
#             df_text = df["sentences "].values
#
#             predictions = []
#             cnt_corrections = 0
#             batch = []
#             for sent in tqdm(df_text):
#                 batch.append(sent.strip().split())
#                 if len(batch) == batch_size:
#                     preds, cnt = model.handle_batch(batch)
#                     predictions.extend(preds)
#                     cnt_corrections += cnt
#                     batch = []
#             if batch:
#                 preds, cnt = model.handle_batch(batch)
#                 predictions.extend(preds)
#                 cnt_corrections += cnt
#
#             result_lines = [" ".join(x) for x in predictions]
#             if to_normalize:
#                 result_lines = [normalize(line) for line in result_lines]
#
#             with open(f"../FINAL_RESULT/gector_result/on_badnet/{dataset_name_i}/{datatype}.txt", 'w') as f:
#                 f.write("\n".join(result_lines) + '\n')
#
#     return 0            ###Return anything is okay



# ## FOR GRAMMATICAL ERRORS
# def predict_for_file(model, batch_size=32, to_normalize=False):
#     gold_file = r"../fce/m2/fce.test.gold.bea19.m2"
#
#     def paragraphs(lines, is_separator=lambda x : x == '\n', joiner=''.join):
#         paragraph = []
#         for line in lines:
#             if is_separator(line):
#                 if paragraph:
#                     yield joiner(paragraph)
#                     paragraph = []
#             else:
#                 paragraph.append(line)
#         if paragraph:
#             yield joiner(paragraph)
#
#     def smart_open(fname, mode = 'r'):
#         if fname.endswith('.gz'):
#             import gzip
#             # Using max compression (9) by default seems to be slow.
#             # Let's try using the fastest.
#             return gzip.open(fname, mode, 1)
#         else:
#             return open(fname, mode)
#
#     def load_annotation(gold_file):
#
#         source_sentences = []
#         perturbation_labels = []
#         gold_edits = []
#         fgold = smart_open(gold_file, 'r')
#         puffer = fgold.read()
#         fgold.close()
#         puffer = puffer.encode('utf8').decode('utf8')
#         for item in paragraphs(puffer.splitlines(True)):
#             item = item.splitlines(False)
#             if item[1] == "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0" or "UNK" in item[1]:
#                 continue
#             sentence = [line[2:].strip() for line in item if line.startswith('S ')]
#             assert sentence != []
#             annotations = {}
#             for line in item[1:]:
#                 if line.startswith('I ') or line.startswith('S '):
#                     continue
#                 assert line.startswith('A ')
#                 line = line[2:]
#                 fields = line.split('|||')
#                 start_offset = int(fields[0].split()[0])
#                 end_offset = int(fields[0].split()[1])
#                 etype = fields[1]
#                 if etype == 'noop' or etype == "UNK" or "ORTH" in etype or "PUNCT" in etype:
#                     start_offset = -1
#                     end_offset = -1
#                 corrections =  [c.strip() if c != '-NONE-' else '' for c in fields[2].split('||')]
#                 # NOTE: start and end are *token* offsets
#                 original = ' '.join(' '.join(sentence).split()[start_offset:end_offset])
#                 annotator = int(fields[5])
#                 if annotator not in annotations.keys():
#                     annotations[annotator] = []
#                 annotations[annotator].append((start_offset, end_offset, original, corrections))
#             tok_offset = 0
#             for this_sentence in sentence:
#                 perturbation_label = [0] * len(this_sentence.split())
#                 tok_offset += len(this_sentence.split())
#                 source_sentences.append(this_sentence)
#                 this_edits = {}
#                 for annotator, annotation in annotations.items():
#                     if annotator == 0:
#                         # this_edits[annotator] = [edit for edit in annotation if edit[0] <= tok_offset and edit[1] <= tok_offset and edit[0] >= 0 and edit[1] >= 0]
#                         for edit in annotation:
#                             if edit[0] <= tok_offset and edit[1] <= tok_offset and edit[0] >= 0 and edit[1] >= 0:
#                                 perturbation_label[edit[0]:edit[1]] = [1] * (edit[1] - edit[0])
#
#                 if len(this_edits) == 0:
#                     this_edits[0] = []
#                 gold_edits.append(this_edits)
#                 perturbation_labels.append(perturbation_label)
#
#         return (source_sentences, perturbation_labels)
#
#
#     source_sentences, gold_edits = load_annotation(gold_file)
#     predictions = []
#     cnt_corrections = 0
#     batch = []
#     for sent,label in tqdm(zip(source_sentences,gold_edits)):
#         if 1 in label:
#             batch.append(sent.split())
#             if len(batch) == batch_size:
#                 preds, cnt = model.handle_batch(batch)
#                 predictions.extend(preds)
#                 cnt_corrections += cnt
#                 batch = []
#     if batch:
#         preds, cnt = model.handle_batch(batch)
#         predictions.extend(preds)
#         cnt_corrections += cnt
#
#     result_lines = [" ".join(x) for x in predictions]
#     if to_normalize:
#         result_lines = [normalize(line) for line in result_lines]
#
#     with open(f"../FINAL_RESULT/gector_result/on_GEC_FCE/test.txt", 'w') as f:
#         f.write("\n".join(result_lines) + '\n')
#     return cnt_corrections



# ## FOR ADVERSARIAL PROMPT
# def predict_for_file(model, batch_size=32, to_normalize=False):
#
#     df = pd.read_csv(
#         f"../promptbench/prompts/adv_prompts.csv")
#
#     for attack_name in ["bertattack", "checklist", "deepwordbug", "stresstest", "textfooler", "textbugger"]:
#         df_attack = df[df["attack name"] == attack_name]
#
#         predictions = []
#         cnt_corrections = 0
#         batch = []
#         for sent in tqdm(df_attack.values):
#             batch.append(sent[2].split())
#             if len(batch) == batch_size:
#                 preds, cnt = model.handle_batch(batch)
#                 predictions.extend(preds)
#                 cnt_corrections += cnt
#                 batch = []
#         if batch:
#             preds, cnt = model.handle_batch(batch)
#             predictions.extend(preds)
#             cnt_corrections += cnt
#
#         result_lines = [" ".join(x) for x in predictions]
#         if to_normalize:
#             result_lines = [normalize(line) for line in result_lines]
#
#         with open(f"../FINAL_RESULT/gector_result/on_prompt/{attack_name}.txt", 'w') as f:
#             f.write("\n".join(result_lines) + '\n')



def main(args):
    # get all paths
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         del_confidence=args.additional_del_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights)

    cnt_corrections = predict_for_file(model,
                                       batch_size=args.batch_size, 
                                       to_normalize=args.normalize)
    # evaluate with m2 or ERRANT
    print(f"Produced overall corrections: {cnt_corrections}")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        default=[r"/root/autodl-tmp/roberta_1_gectorv2.th"])        # The model is saved locally
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=512)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
                                 'bert-large', 'roberta-large', 'xlnet-large'],
                        help='Name of the transformer model.',
                        default=r'/root/autodl-tmp/roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0.0)
    parser.add_argument('--additional_del_confidence',
                        type=float,
                        help='How many probability to add to $DELETE token.',
                        default=0.0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='Minimum probability for each action to apply. '
                             'Also, minimum error probability, as described in the paper.',
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--normalize',
                        help='Use for text simplification.',
                        action='store_true')
    args = parser.parse_args()
    main(args)
