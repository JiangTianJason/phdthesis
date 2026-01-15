###RMSE distance，Table 6.12

import numpy as np
import pandas as pd


def calculate_rmse(points):
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse

for dataset in ["sst-2", "yelp","agnews"]:
    number = 1 if dataset in ["sst-2", "yelp"] else 0

    # ##poison_detection
    # loaded_text_label = pd.read_csv(fr"../BadActs-master/poison_data/{dataset}/{number}/stylebkd/test-detect.csv",index_col=0)
    # stylebkd_poison_indices = loaded_text_label.loc[loaded_text_label["2"] == 1].index
    # loaded_text_label = pd.read_csv(fr"../BadActs-master/poison_data/{dataset}/{number}/synbkd/test-detect.csv",index_col=0)
    # synbkd_poison_indices = loaded_text_label.loc[loaded_text_label["2"] == 1].index
    # print(len(stylebkd_poison_indices) == len(synbkd_poison_indices))
    # stylebkd = np.load(fr"../BadActs-master/embed_result/stylebkd/{dataset}/test_detect_embedding_batch.npy")[stylebkd_poison_indices]
    # synbkd = np.load(fr"../BadActs-master/embed_result/synbkd/{dataset}/test_detect_embedding_batch.npy")[synbkd_poison_indices]


    # ##poison_filter
    # loaded_text_label = pd.read_csv(fr"../BadActs-master/poison_data/{dataset}/{number}/stylebkd/dirty/0.2/train-poison.csv",index_col=0)
    # stylebkd_poison_indices = loaded_text_label.loc[loaded_text_label["2"] == 1].index
    # loaded_text_label = pd.read_csv(fr"../BadActs-master/poison_data/{dataset}/{number}/synbkd/dirty/0.2/train-poison.csv",index_col=0)
    # synbkd_poison_indices = loaded_text_label.loc[loaded_text_label["2"] == 1].index
    # print(len(stylebkd_poison_indices) == len(synbkd_poison_indices))
    # stylebkd = np.load(fr"../BadActs-master/embed_result/stylebkd/{dataset}/train_poison_embedding_batch.npy")[stylebkd_poison_indices]
    # synbkd = np.load(fr"../BadActs-master/embed_result/synbkd/{dataset}/train_poison_embedding_batch.npy")[synbkd_poison_indices]


    ###adversarial
    ##train
    stylebkd_train_all = pd.read_csv(fr"./Distinguishing-Non-Natural-main/src/data_corresponding_detection/{dataset}/stylebkd/train.tsv",index_col=0,sep="\t")
    synbkd_train_all = pd.read_csv(fr"./Distinguishing-Non-Natural-main/src/data_corresponding_detection/{dataset}/synbkd/train.tsv",index_col=0,sep="\t")

    overlap_train_index = stylebkd_train_all.index.intersection(synbkd_train_all.index)
    stylebkd_train = np.load(fr"../DeepStyle-master/BadActs_embed_result/stylebkd/{dataset}/train_poison_embedding_batch.npy")[overlap_train_index]
    synbkd_train = np.load(fr"../DeepStyle-master/BadActs_embed_result/synbkd/{dataset}/train_poison_embedding_batch.npy")[overlap_train_index]

    ##test
    stylebkd_test_all = pd.read_csv(fr"./Distinguishing-Non-Natural-main/src/data_corresponding_detection/{dataset}/stylebkd/test_non_shuffle.tsv",index_col=0,sep="\t")
    synbkd_test_all = pd.read_csv(fr"./Distinguishing-Non-Natural-main/src/data_corresponding_detection/{dataset}/synbkd/test_non_shuffle.tsv",index_col=0,sep="\t")
    benign = pd.read_csv(fr"../BadActs-master/poison_data/{dataset}/{number}/stylebkd/test-detect.csv",index_col=0)

    overlap_test_index = stylebkd_test_all.index.intersection(synbkd_test_all.index)
    benign_count = len(benign[benign["2"] == 0]) - 1
    overlap_test_index += benign_count
    stylebkd_test = np.load(fr"../DeepStyle-master/BadActs_embed_result/stylebkd/{dataset}/test_detect_embedding_batch.npy")[overlap_test_index]
    synbkd_test = np.load(fr"../DeepStyle-master/BadActs_embed_result/synbkd/{dataset}/test_detect_embedding_batch.npy")[overlap_test_index]

    stylebkd = np.concatenate((stylebkd_train,stylebkd_test))
    synbkd = np.concatenate((synbkd_train,synbkd_test))


    ###Public Area###
    dev_clean_dimension_reduced = np.concatenate((stylebkd,synbkd))

    print("adversarial",dataset,"average_distance")
    stylebkd_eigenvalues = calculate_rmse(dev_clean_dimension_reduced[:len(stylebkd)])
    print(stylebkd_eigenvalues)

    synbkd_eigenvalues = calculate_rmse(dev_clean_dimension_reduced[len(stylebkd):])
    print(synbkd_eigenvalues)

    with open(fr"adversarial/{dataset}_result.txt","a") as f:
        f.write(" \n")
        f.write("stylebkd_eigenvalues:  {}\n".format(stylebkd_eigenvalues))
        f.write("synbkd_eigenvalues:  {}\n".format(synbkd_eigenvalues))