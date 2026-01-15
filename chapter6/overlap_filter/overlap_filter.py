import pandas as pd

###poison_filter###
for dataset in ["sst-2","yelp","agnews"]:
    number = 0 if dataset == "agnews" else 1

    stylebkd_reference_all = pd.read_csv(fr"D:\Download\Sentence_Level_defense\BadActs-master\poison_data\{dataset}\{number}\stylebkd\train-poison.csv",index_col=0)
    synbkd_reference_all = pd.read_csv(fr"D:\Download\Sentence_Level_defense\BadActs-master\poison_data\{dataset}\{number}\synbkd\train-poison.csv",index_col=0)

    stylebkd_all = pd.read_csv(fr"D:\Download\Sentence_Level_defense\BadActs-master\poison_data\{dataset}\{number}\stylebkd\dirty\0.2\train-poison.csv",index_col=0)
    synbkd_all = pd.read_csv(fr"D:\Download\Sentence_Level_defense\BadActs-master\poison_data\{dataset}\{number}\synbkd\dirty\0.2\train-poison.csv",index_col=0)
    clean_all = pd.read_csv(fr"D:\Download\Sentence_Level_defense\BadActs-master\poison_data\{dataset}\{number}\stylebkd\train-clean.csv",index_col=0)

    stylebkd = stylebkd_all[stylebkd_all["2"] == 1]["0"]
    synbkd = synbkd_all[synbkd_all["2"] == 1]["0"]

    clean_index = []
    for i,j in zip(stylebkd,synbkd):
        a = stylebkd_reference_all[stylebkd_reference_all["0"] == i].index
        clean_index.append(clean_all.iloc[a].index.tolist()[0])
    clean = clean_all.iloc[clean_index]["0"].reset_index()
    stylebkd = stylebkd.reset_index()
    synbkd = synbkd.reset_index()
    del stylebkd["index"]
    del synbkd["index"]
    del clean["index"]

    output = pd.concat([clean, stylebkd, synbkd],axis=1)

    output.to_csv(fr"poison_filter/{dataset}.csv",header=["original","stylebkd","synbkd"])


# ###poison_detection###
# for dataset in ["sst-2","yelp","agnews"]:
#     number = 0 if dataset == "agnews" else 1
#
#     stylebkd_all = pd.read_csv(fr"D:\Download\Sentence_Level_defense\BadActs-master\poison_data\{dataset}\{number}\stylebkd\test-detect.csv",index_col=0)
#     synbkd_all = pd.read_csv(fr"D:\Download\Sentence_Level_defense\BadActs-master\poison_data\{dataset}\{number}\synbkd\test-detect.csv",index_col=0)
#     clean_all = pd.read_csv(fr"D:\Download\Sentence_Level_defense\BadActs-master\poison_data\{dataset}\{number}\stylebkd\test-clean.csv",index_col=0)
#
#     stylebkd = stylebkd_all[stylebkd_all["2"] == 1]["0"].reset_index()
#     del stylebkd['index']
#     synbkd = synbkd_all[synbkd_all["2"] == 1]["0"].reset_index()
#     del synbkd['index']
#     clean = clean_all[clean_all["1"] != number]["0"].reset_index()
#     del clean['index']
#
#     output = pd.concat([clean,stylebkd,synbkd],axis=1)
#
#     output.to_csv(fr"poison_detection/{dataset}.csv",header=["original","stylebkd","synbkd"])


# ###adversarial###
# for dataset in ["sst-2","yelp","agnews"]:
#     number = 0 if dataset == "agnews" else 1
#
#     ###train
#     stylebkd_train_all = pd.read_csv(fr"D:\Download\TextAttack-0.3.8\Distinguishing-Non-Natural-main\src\data_corresponding_detection\{dataset}\stylebkd\train.tsv",index_col=0,sep="\t")
#     synbkd_train_all = pd.read_csv(fr"D:\Download\TextAttack-0.3.8\Distinguishing-Non-Natural-main\src\data_corresponding_detection\{dataset}\synbkd\train.tsv",index_col=0,sep="\t")
#
#     overlap_index = stylebkd_train_all.index.intersection(synbkd_train_all.index)
#
#     stylebkd_train_overlap = stylebkd_train_all[:int(len(stylebkd_train_all) / 2)].loc[overlap_index]
#     synbkd_train_overlap = synbkd_train_all[:int(len(synbkd_train_all) / 2)].loc[overlap_index]
#     clean_train_overlap = stylebkd_train_all[int(len(stylebkd_train_all) / 2):].loc[overlap_index]
#
#     stylebkd_train = stylebkd_train_overlap["sentence"].reset_index()
#     del stylebkd_train['index']
#     synbkd_train = synbkd_train_overlap["sentence"].reset_index()
#     del synbkd_train['index']
#     clean_train = clean_train_overlap["sentence"].reset_index()
#     del clean_train['index']
#
#     ###test
#     stylebkd_test_all = pd.read_csv(fr"D:\Download\TextAttack-0.3.8\Distinguishing-Non-Natural-main\src\data_corresponding_detection\{dataset}\stylebkd\test_non_shuffle.tsv",index_col=0,sep="\t")
#     synbkd_test_all = pd.read_csv(fr"D:\Download\TextAttack-0.3.8\Distinguishing-Non-Natural-main\src\data_corresponding_detection\{dataset}\synbkd\test_non_shuffle.tsv",index_col=0,sep="\t")
#
#     overlap_index = stylebkd_test_all.index.intersection(synbkd_test_all.index)
#
#     stylebkd_test_overlap = stylebkd_test_all[:int(len(stylebkd_test_all) / 2)].loc[overlap_index]
#     synbkd_test_overlap = synbkd_test_all[:int(len(synbkd_test_all) / 2)].loc[overlap_index]
#     clean_test_overlap = stylebkd_test_all[int(len(stylebkd_test_all) / 2):].loc[overlap_index]
#
#     stylebkd_test = stylebkd_test_overlap["sentence"].reset_index()
#     del stylebkd_test['index']
#     synbkd_test = synbkd_test_overlap["sentence"].reset_index()
#     del synbkd_test['index']
#     clean_test = clean_test_overlap["sentence"].reset_index()
#     del clean_test['index']
#
#     stylebkd = pd.concat([stylebkd_train,stylebkd_test])
#     synbkd = pd.concat([synbkd_train,synbkd_test])
#     clean = pd.concat([clean_train,clean_test])
#
#     output = pd.concat([clean,stylebkd,synbkd],axis=1)
#
#     output.to_csv(fr"adversarial/{dataset}.csv",header=["original","stylebkd","synbkd"])