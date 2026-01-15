### Prepare stylistic embedding before the defence, and you can embed any sentence using the LUAR model.

import math
from typing import List, Union
import pandas as pd
import numpy as np

np.random.seed(42)
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, pipeline
from tqdm import tqdm


####No need to comment, publicly used
device = torch.device("cpu")
model = AutoModel.from_pretrained(r"LUAR",trust_remote_code=True)       ###The model is saved locally
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(r"LUAR")

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



# ## FOR BACKDOOR
# for dataset_name in ["agnews"]:
#     for attack_method in ["synbkd"]:
#         for datatype in ["train-poison"]:
#             if dataset_name in ["yelp", "sst-2"]:
#                 number = 1
#             else:
#                 number = 0
#             test_detect_path = fr"./BadActs/poison_data/{dataset_name}/{number}/{attack_method}/{datatype}.csv"
#             data = pd.read_csv(test_detect_path,index_col=0).values
#             test_detect_data = [(d[0]) for d in data]
#
#             embedding_list = []
#             # Sample document:
#             for text in tqdm(test_detect_data):
#                 embedding_list.append(get_uar_embedding(text)[0].numpy())
#
#             np.save(f'./embed_result/BadActs_embed_result/{attack_method}/{dataset_name}/{datatype}_embedding_batch.npy',np.array(embedding_list))
#             print("Done ->",f'./embed_result/BadActs_embed_result/{attack_method}/{dataset_name}/{datatype}_embedding_batch.npy')



# ## FOR JAILBREAK
# for i in ["train","test"]:
#     test_detect_path = fr"./jailbreak_llms-main/data/balanced_jailbreak_dataset_{i}_balanced.csv"
#     test_data = pd.read_csv(test_detect_path,sep=",").values[:,0]
#
#     embedding_list = []
#     # Sample document:
#     for text in tqdm(test_data):
#         embedding_list.append(get_uar_embedding(text)[0].numpy())
#
#     np.save(f'./embed_result/Jailbreak_embed_result/balanced_jailbreak_dataset_{i}_embedding_batch.npy', np.array(embedding_list))
#     print("Done ->", f'./embed_result/Jailbreak_embed_result/balanced_jailbreak_dataset_{i}_embedding_batch.npy')



# ## FOR ADVERSARIAL
label2id = {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2,
    "LABEL_3": 3
}

for dataset_name in ["sst-2","yelp","agnews"]:
    number = 1 if dataset_name in ["sst-2","yelp"] else 0
    for attack_name in ["stylebkd","synbkd"]:
        for datatype in ["test-poison"]:
            model_path = fr"bert-base-uncased-{dataset_name}"     ### the model is saved locally
            data_path = fr"./BadActs/poison_data/{dataset_name}/{number}/{attack_name}/{datatype}.csv"
            all_data = pd.read_csv(data_path,index_col=0,sep=",")

            origin_path = fr"./BadActs/poison_data/{dataset_name}/{number}/{attack_name}/test-clean.csv"
            origin_data = pd.read_csv(origin_path,index_col=0,sep=",")
            origin_data = origin_data[origin_data['1'] != number].values

            if len(all_data.values) != len(origin_data):
                print("Skip!!!",dataset_name,attack_name)
                continue

            datatype = datatype.replace("-", "_")

            if "poison" in datatype:
                classifier = pipeline("text-classification",model=model_path,truncation=True)

                status,embedding_list,label_list = [],[],[]
                for idx,i in tqdm(enumerate(all_data.values)):
                    text = i[0]

                    label = origin_data[idx][1]
                    label_list.append(label)

                    output = classifier(text)
                    if label2id[output[0]["label"]] != label:
                        status.append("Successful")
                    else:
                        status.append("Failed")

                    # embedding_list.append(get_uar_embedding(text)[0].numpy())

                all_data["label"] = label_list
                all_data["status"] = status
                all_data.to_csv(fr"./embed_result/BadActs_embed_result/{attack_name}/{dataset_name}/{datatype}.tsv",sep=",",index=False)
                np.save(fr'./embed_result/BadActs_embed_result/{attack_name}/{dataset_name}/{datatype}_embedding_batch.npy', np.array(embedding_list))
                print("Saved !!!——>",fr"./embed_result/BadActs_embed_result/{attack_name}/{dataset_name}/{datatype}_embedding_batch.npy")

            # else:
            #     status,embedding_list,label_list = [],[],[]
            #     for idx, i in tqdm(enumerate(all_data.values)):
            #         text = i[0]
            #         embedding_list.append(get_uar_embedding(text)[0].numpy())
            #
            #     # np.save(
            #     #     fr'./BadActs_embed_result/{attack_name}/{dataset_name}/{datatype}_embedding_batch.npy',
            #     #     np.array(embedding_list))
            #     # print("Saved !!!——>",
            #     #       fr"./BadActs_embed_result/{attack_name}/{dataset_name}/{datatype}_embedding_batch.npy")


dataset_name = "agnews"
if dataset_name in ["yelp", "sst-2"]:
    number = 1
else:
    number = 0
for attack_method in ["stylebkd"]:
    test_detect_path = fr"./Distinguishing-Non-Natural-main/src/data_detection/{dataset_name}/{attack_method}/train.tsv"
    data = pd.read_csv(test_detect_path,index_col=0,sep="\t").dropna().values
    test_detect_data = [(d[0]) for d in data]

    embedding_list = []

    try:
        for text in tqdm(test_detect_data):
            embedding_list.append(get_uar_embedding(text)[0].numpy())
        np.save(fr'./Distinguishing-Non-Natural-main/src/data_detection/{dataset_name}/{attack_method}/train_embedding_batch.npy', embedding_list)
        print("Done ->",fr'./Distinguishing-Non-Natural-main/src/data_detection/{dataset_name}/{attack_method}/train_embedding_batch.npy')
    except:
        np.save(fr'./Distinguishing-Non-Natural-main/src/data_detection/{dataset_name}/{attack_method}/train_embedding_batch.npy', embedding_list)
        print("Done ->",fr'./Distinguishing-Non-Natural-main/src/data_detection/{dataset_name}/{attack_method}/train_embedding_batch.npy')
        print("len(embedding_list): ",len(embedding_list))