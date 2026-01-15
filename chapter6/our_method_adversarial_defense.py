###（Strictly corresponding examples in the dataset，Table 6.5、6.6，Figure 6.6）Our method： "SVC"

from sklearn.metrics import classification_report
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import umap

for attack_method in ["synbkd"]:
    for dataset_name in ["agnews"]:

        ###Especially for the "stylebkd+agnews" combination
        # loaded_test = pd.read_csv(
        #     fr"./Distinguishing-Non-Natural-main/src/data_corresponding_detection/{dataset_name}/{attack_method}/test_non_shuffle.tsv",
        #     index_col=0, sep="\t")
        # test_labels = loaded_test["label"].values
        # test_embedding = np.load(fr"./Distinguishing-Non-Natural-main/src/data_corresponding_detection/{dataset_name}/{attack_method}/test_non_shuffle_embedding_batch.npy")

        loaded_test = pd.read_csv(
            fr"./Distinguishing-Non-Natural-main/src/data_corresponding_detection/{dataset_name}/{attack_method}/test.tsv",
            index_col=0, sep="\t")
        test_labels = loaded_test["label"].values
        test_embedding = np.load(fr"./Distinguishing-Non-Natural-main/src/data_corresponding_detection/{dataset_name}/{attack_method}/test_embedding_batch.npy")

        loaded_train = pd.read_csv(
            fr"./Distinguishing-Non-Natural-main/src/data_corresponding_detection/{dataset_name}/{attack_method}/train.tsv",
            index_col=0, sep="\t")
        train_labels = loaded_train["label"].values
        train_embedding = np.load(fr"./Distinguishing-Non-Natural-main/src/data_corresponding_detection/{dataset_name}/{attack_method}/train_embedding_batch.npy")

        mapper = umap.UMAP(n_components=2,random_state=42,transform_seed=42)
        train_dimension_reduced = mapper.fit_transform(train_embedding)
        test_dimension_reduced = mapper.transform(test_embedding)

        outlier_detection_model = SVC(random_state=42)
        outlier_detection_model.fit(train_embedding,train_labels)

        predictions = outlier_detection_model.predict(test_embedding)

        report = classification_report(test_labels,predictions,digits=5)

        with open('./Distinguishing-Non-Natural-main/detection_corresponding_results/our_method.txt', 'a') as f:
            print("{}_{}".format(attack_method,dataset_name),file = f)
            print(report + "\n", file=f)

        train_clean = np.where(train_labels == 0)
        test_clean = np.where(test_labels == 0)
        train_adversarial = np.where(train_labels == 1)
        test_adversarial = np.where(test_labels == 1)

        plt.figure(figsize=(10, 8), dpi=300)
        plt.scatter(test_dimension_reduced[test_clean][:,0],test_dimension_reduced[test_clean][:,1],s=1.0,label="test_clean")
        plt.scatter(test_dimension_reduced[test_adversarial][:,0],test_dimension_reduced[test_adversarial][:,1],s=1.0,label="test_adversarial")
        plt.scatter(train_dimension_reduced[train_clean][:,0],train_dimension_reduced[train_clean][:,1],s=1.0,label="train_clean")
        plt.scatter(train_dimension_reduced[train_adversarial][:,0],train_dimension_reduced[train_adversarial][:,1],s=1.0,label="train_adversarial")

        plt.title(f'{dataset_name}_{attack_method}',fontsize=15)
        plt.xlabel('UMAP Dimension 1', fontsize=15)
        plt.ylabel('UMAP Dimension 2', fontsize=15)

        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)

        plt.legend(markerscale=1.5,fontsize=15)
        plt.savefig(fr"./Distinguishing-Non-Natural-main/saved_figure_corresponding/{dataset_name}_{attack_method}.jpg", dpi=300)
        plt.show()



###（Non-Strictly corresponding examples in the dataset，Table 6.7、6.8，Figure 6.7）Our method： "SVC"

from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.svm import SVC

for attack_method in ["stylebkd","synbkd"]:
    for dataset_name in ["agnews","sst-2","yelp"]:
        number = 1 if dataset_name in ["sst-2","yelp"] else 0
        label = []

        loaded_text_label = pd.read_csv(
            fr"./BadActs/poison_data/{dataset_name}/{number}/{attack_method}/test-detect.csv",
            index_col=0)
        clean_indices = loaded_text_label.loc[loaded_text_label["2"] == 0].index
        loaded_embeddings_test_detect = np.load(fr"./embed_result/BadActs_embed_result/{attack_method}/{dataset_name}/test_detect_embedding_batch.npy")[clean_indices]

        loaded_text_label_train_clean = pd.read_csv(fr"./embed_result/BadActs_embed_result/{attack_method}/{dataset_name}/test_poison.tsv",sep=",")
        success_indices_bible = loaded_text_label_train_clean.loc[loaded_text_label_train_clean["status"] == "Successful"].index
        loaded_embeddings_test = np.load(fr'./embed_result/BadActs_embed_result/{attack_method}/{dataset_name}/test_detect_embedding_batch.npy')[len(clean_indices):]
        loaded_embeddings_test = loaded_embeddings_test[success_indices_bible]

        clean_adversarial_test = np.concatenate((loaded_embeddings_test_detect,loaded_embeddings_test))

        def load_embedding(dataset_type):
            loaded_embeddings = np.load(fr"./Distinguishing-Non-Natural-main/src/data_detection/{dataset_name}/{attack_method}/train_embedding_batch.npy")
            labels = pd.read_csv(fr"./Distinguishing-Non-Natural-main/src/data_detection/{dataset_name}/{attack_method}/train.tsv",sep="\t").dropna()
            labels = [i for i in labels.values[:,2]]
            return loaded_embeddings,np.array(labels)

        train_embedding,train_label = load_embedding("train")
        train_clean = np.where(train_label == 0)
        train_adversarial = np.where(train_label == 1)

        mapper = umap.UMAP(n_components=2,random_state=42,transform_seed=42)
        dev_clean_dimension_reduced = mapper.fit_transform(train_embedding)
        test_adversarial_dimension_reduced = mapper.transform(clean_adversarial_test)

        outlier_detection_model = SVC(random_state=42)
        outlier_detection_model.fit(train_embedding,train_label)

        predictions = outlier_detection_model.predict(clean_adversarial_test)
        poison_labels = [0] * len(loaded_embeddings_test_detect) + [1] * len(loaded_embeddings_test)

        report = classification_report(poison_labels,predictions,digits=5)

        with open('./Distinguishing-Non-Natural-main/detection_results/our_method.txt', 'a') as f:
            print("{}_{}".format(attack_method,dataset_name),file = f)
            print(report + "\n", file=f)

        plt.figure(figsize=(10, 8), dpi=300)
        plt.scatter(test_adversarial_dimension_reduced[:len(loaded_embeddings_test_detect)][:,0],test_adversarial_dimension_reduced[:len(loaded_embeddings_test_detect)][:,1],s=1.0,label="test_clean")
        plt.scatter(test_adversarial_dimension_reduced[len(loaded_embeddings_test_detect):][:,0],test_adversarial_dimension_reduced[len(loaded_embeddings_test_detect):][:,1],s=1.0,label="test_adversarial")
        plt.scatter(dev_clean_dimension_reduced[train_clean][:,0],dev_clean_dimension_reduced[train_clean][:,1],s=1.0,label="train_clean")
        plt.scatter(dev_clean_dimension_reduced[train_adversarial][:,0],dev_clean_dimension_reduced[train_adversarial][:,1],s=1.0,label="train_adversarial")

        plt.title(f'{dataset_name}_{attack_method}',fontsize=15)
        plt.xlabel('UMAP Dimension 1', fontsize=15)
        plt.ylabel('UMAP Dimension 2', fontsize=15)

        plt.tick_params(axis='x', labelsize=14)
        plt.tick_params(axis='y', labelsize=14)

        plt.legend(markerscale=1.5,fontsize=15)
        plt.savefig(fr"./Distinguishing-Non-Natural-main/saved_figure/{dataset_name}_{attack_method}.jpg", dpi=300)



####（Strictly corresponding examples in the dataset，Table 6.5、6.6）Comparison method：Distinguishing-Non-Natural-main
#### Failed attacks are amplified, while successful attacks are enhanced.
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

attack_method = "randomization_synbkd"
dataset_name = "yelp"

loaded_text_label = pd.read_csv(
    fr"./Distinguishing-Non-Natural-main/src/data_corresponding_detection/{dataset_name}/synbkd/test.tsv",
    index_col=0,sep="\t")
texts = loaded_text_label["sentence"].values.tolist()
labels = loaded_text_label["label"].values.tolist()

tokenizer = BertTokenizer.from_pretrained(
    fr"./Distinguishing-Non-Natural-main/src/experiments_detector_corresponding/{dataset_name}/{attack_method}")
model = BertForSequenceClassification.from_pretrained(
    fr"./Distinguishing-Non-Natural-main/src/experiments_detector_corresponding/{dataset_name}/{attack_method}")

predictions = []
for i in tqdm(texts):
    inputs = tokenizer(i, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predictions.append(logits.argmax().item())

with open(fr"./Distinguishing-Non-Natural-main/src/data_corresponding_detection/{dataset_name}/test_{attack_method}_result.txt","a") as f:
    f.write(classification_report(labels, predictions, digits=5))
    print("output to ->",fr"./Distinguishing-Non-Natural-main/src/data_corresponding_detection/{dataset_name}/test_{attack_method}_result.txt")



####（Non-Strictly corresponding examples in the dataset，Table 6.7、6.8）Comparison method：Distinguishing-Non-Natural-main
#### Failed attacks are amplified, while successful attacks are enhanced.
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

attack_method = "synbkd"
dataset_name = "sst-2"
number = 1 if dataset_name in ["sst-2", "yelp"] else 0

loaded_text_label = pd.read_csv(
    fr"./BadActs/poison_data/{dataset_name}/{number}/{attack_method}/test-detect.csv",
    index_col=0)
clean_indices = loaded_text_label[loaded_text_label["2"] == 0]["0"].values.tolist()

loaded_text_label_train_clean = pd.read_csv(
    fr"./embed_result/BadActs_embed_result/{attack_method}/{dataset_name}/test_poison.tsv",
    sep=",")
success_indices_bible = loaded_text_label_train_clean[loaded_text_label_train_clean["status"] == "Successful"][
    "0"].values.tolist()

text = clean_indices + success_indices_bible
tokenizer = BertTokenizer.from_pretrained(
    fr"./Distinguishing-Non-Natural-main/src/experiments_detector/{dataset_name}/{attack_method}")
model = BertForSequenceClassification.from_pretrained(
    fr"./Distinguishing-Non-Natural-main/src/experiments_detector/{dataset_name}/{attack_method}")

predictions = []
for i in tqdm(text):
    inputs = tokenizer(i, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predictions.append(logits.argmax().item())

poison_labels = [0] * len(clean_indices) + [1] * len(success_indices_bible)

print(classification_report(poison_labels, predictions, digits=5))