###Our method

import os
from .defender import Defender
from sklearn.metrics import roc_auc_score
from openbackdoor.victims import Victim
from typing import *
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from openbackdoor.utils import evaluate_detection, logger
from transformers import AutoModel, AutoTokenizer,AutoConfig
import math
from typing import List, Union
import torch.nn.functional as F
import umap
from sklearn.cluster import KMeans
import pandas as pd


def calculate_auroc(scores, labels):
    scores = [-s for s in scores]
    auroc = roc_auc_score(labels, scores)
    return auroc


def draw_distribution_isolationforest(extracted_embeddings_dev_clean,extracted_embeddings_poison_data,attack_method,dataset_name,poison_labels):
    mapper = umap.UMAP(n_components=2, random_state=42, transform_seed=42)

    concate = np.concatenate((extracted_embeddings_dev_clean,extracted_embeddings_poison_data))
    dimension_reduced = mapper.fit_transform(concate)

    plt.figure(figsize=(10, 8))
    plt.scatter(dimension_reduced[len(extracted_embeddings_dev_clean):len(extracted_embeddings_dev_clean) + poison_labels.count(0), 0],
                dimension_reduced[len(extracted_embeddings_dev_clean):len(extracted_embeddings_dev_clean) + poison_labels.count(0), 1], c="#4daf4a", s=1.0,
                label="Non-Poisoned")
    plt.scatter(dimension_reduced[-poison_labels.count(1):, 0],
                dimension_reduced[-poison_labels.count(1):, 1], c="#8B0000", s=2.0,
                label="Poisoned")
    plt.scatter(dimension_reduced[:len(extracted_embeddings_dev_clean), 0], dimension_reduced[:len(extracted_embeddings_dev_clean), 1], c='#0000FF', s=1.0,
                label="Dev-Clean")
    plt.title(f'{dataset_name}_{attack_method}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    plt.legend()
    plt.savefig(fr"./saved_figure/detection/{dataset_name}_{attack_method}.jpg", dpi=300)
    print("Saved——>",
                fr"./saved_figure/detection/{dataset_name}_{attack_method}.jpg")


def draw_distribution_kmeans(extracted_embeddings_train_poison,attack_method,dataset_name,poison_label):
    non_poisoned_end = poison_label.count(0)
    mapper = umap.UMAP(n_components=2, random_state=42, transform_seed=42)

    dimension_reduced = mapper.fit_transform(extracted_embeddings_train_poison)

    plt.figure(figsize=(10, 8))
    plt.scatter(dimension_reduced[:non_poisoned_end, 0],
                dimension_reduced[:non_poisoned_end, 1], c="#4daf4a", s=1.0,
                label="Non-Poisoned")
    plt.scatter(dimension_reduced[non_poisoned_end:, 0],
                dimension_reduced[non_poisoned_end:, 1], c="#8B0000", s=2.0,
                label="Poisoned")
    plt.title(f'{dataset_name}_{attack_method}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    plt.legend()
    plt.savefig(fr"./saved_figure/correction/{dataset_name}_{attack_method}.jpg", dpi=300)
    print("Saved——>",
                fr"./saved_figure/correction/{dataset_name}_{attack_method}.jpg")


def save_embedding(embedding,dataset_name,attack_method,datatype):
    np.save(f'../embed_result/{attack_method}/{dataset_name}/{datatype}_embedding_batch.npy', embedding)
    print("!!!Saved embedding to ->",f'../embed_result/{attack_method}/{dataset_name}/{datatype}_embedding_batch.npy')


device = torch.device("cuda:0")
LUAR_model = AutoModel.from_pretrained(r"LUAR",trust_remote_code=True)       ###The model is saved locally
LUAR_model.to(device)
LUAR_model.eval()
LUAR_tokenizer = AutoTokenizer.from_pretrained(r"LUAR")       ###The model is saved locally


class Stylistic_Defender(Defender):

    def __init__(
            self,
            victim: Optional[str] = 'bert',
            frr: Optional[float] = 0.05,
            poison_dataset: Optional[str] = 'sst-2',
            attacker: Optional[str] = 'badnets',
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.frr = frr
        self.victim = victim
        self.poison_dataset = poison_dataset
        self.attacker = attacker


    def detect(
            self,
            model: Victim,
            clean_data: List,
            poison_data: List,
    ):
        clean_dev_ = clean_data["dev"]
        random.seed(2024)
        logger.info("Use {} clean dev data, {} poisoned test data in total".format(len(clean_dev_), len(poison_data)))
        poison_labels = [d[2] for d in poison_data]
        poison_text = [d[0] for d in poison_data]

        extracted_embeddings_dev_clean = []
        for idx, (text, label, poison_label) in enumerate(clean_dev_):
            extracted_embeddings_dev_clean.append(self.get_uar_embedding(text)[0].cpu().numpy())

        save_embedding(np.array(extracted_embeddings_dev_clean),self.poison_dataset,self.attacker,"dev_clean")

        outlier_detection_model = IsolationForest(contamination=self.frr, random_state=42)              ###set "contamination" == FRR，described in page 88
        outlier_detection_model.fit(np.array(extracted_embeddings_dev_clean))

        preds,poison_scores,extracted_embeddings_poison_data = [],[],[]
        for text in poison_text:
            extracted_embedding = self.get_uar_embedding(text).cpu().numpy()
            extracted_embeddings_poison_data.append(extracted_embedding[0])
            poison_scores.append(outlier_detection_model.decision_function(extracted_embedding))
            prediction = 1 if outlier_detection_model.predict(extracted_embedding) == -1 else 0
            preds.append(prediction)

        save_embedding(np.array(extracted_embeddings_poison_data),self.poison_dataset,self.attacker,"test_detect")

        auroc = calculate_auroc(poison_scores, poison_labels)
        logger.info("auroc: {}".format(auroc))

        logger.info("Constrain FRR to {}".format(self.frr))

        draw_distribution_isolationforest(extracted_embeddings_dev_clean,extracted_embeddings_poison_data,self.attacker,self.poison_dataset,poison_labels)

        return np.array(preds), auroc


    def correct(
        self,
        poison_data: List,
        clean_data: Optional[List] = None,
        model: Optional[Victim] = None
    ):
        poison_label = [d[2] for d in poison_data]

        if not os.path.exists(r"../embed_result/{}/{}/train_poison_embedding_batch.npy".format(self.attacker,self.poison_dataset)):
            extracted_embeddings_train_poison = np.array([self.get_uar_embedding(d[0])[0].cpu().numpy() for d in poison_data])
            save_embedding(extracted_embeddings_train_poison,self.poison_dataset,self.attacker,"train_poison")
        else:
            extracted_embeddings_train_poison = np.load(r"../embed_result/{}/{}/train_poison_embedding_batch.npy".format(self.attacker,self.poison_dataset))

        kmeans_model = KMeans(random_state=42)
        predictions = kmeans_model.fit_predict(extracted_embeddings_train_poison)

        draw_distribution_kmeans(extracted_embeddings_train_poison,self.attacker,self.poison_dataset,poison_label)

        filtered_dataset = self.filtering(poison_data,poison_label,predictions)

        return filtered_dataset


    def filtering(self, dataset: List, y_true: List, y_pred: List):     ###Use the same "filtering" function in CUBE

        logger.info("Filtering suspicious samples")

        dropped_indices = []
        if isinstance(y_true[0], torch.Tensor):
            y_true = [y.item() for y in y_true]

        for true_label in set(y_true):

            groundtruth_samples = np.where(y_true == true_label * np.ones_like(y_true))[0]

            drop_scale = 0.5 * len(groundtruth_samples)

            # Check the predictions for samples of this groundtruth label
            predictions = set()
            for i, pred in enumerate(y_pred):
                if i in groundtruth_samples:
                    predictions.add(pred)

            if len(predictions) > 1:
                count = pd.DataFrame(columns=['predictions'])

                for pred_label in predictions:
                    count.loc[pred_label, 'predictions'] = \
                        np.sum(np.where((y_true == true_label * np.ones_like(y_true)) * \
                                        (y_pred == pred_label * np.ones_like(y_pred)),
                                        np.ones_like(y_pred), np.zeros_like(y_pred)))
                cluster_order = count.sort_values(by='predictions', ascending=True)

                # we always preserve the largest prediction cluster
                for pred_label in cluster_order.index.values[:-1]:
                    item = cluster_order.loc[pred_label, 'predictions']
                    if item < drop_scale:
                        idx = np.where((y_true == true_label * np.ones_like(y_true)) * \
                                       (y_pred == pred_label * np.ones_like(y_pred)))[0].tolist()

                        dropped_indices.extend(idx)

        filtered_dataset = []
        for i, data in enumerate(dataset):
            if i not in dropped_indices:
                filtered_dataset.append(data)

        return filtered_dataset


    def get_uar_embedding(self,sample: Union[List[str], str]):
        if isinstance(sample, str):
            sample = [sample]

        tok = LUAR_tokenizer.batch_encode_plus(
            sample,
            truncation=False,
            padding=True,
            return_tensors="pt"
        )

        # UAR's backbone can handle up to 512 tokens
        # Here we're padding the sample to the nearest multiple of 512:
        _, NT = tok["input_ids"].size()
        nearest = 512 * int(math.ceil(NT / 512))
        tok["input_ids"] = F.pad(tok["input_ids"], (1, nearest - NT - 1), value=LUAR_tokenizer.pad_token_id)
        tok["attention_mask"] = F.pad(tok["attention_mask"], (1, nearest - NT - 1), value=0)

        # Reshape into (batch_size=1, history_size=N, num_tokens=512)
        tok["input_ids"] = tok["input_ids"].reshape(1, -1, 512).to(device)
        tok["attention_mask"] = tok["attention_mask"].reshape(1, -1, 512).to(device)

        with torch.inference_mode():
            out = LUAR_model(**tok)
            out = F.normalize(out, p=2.0)
        return out