import csv
import pandas
import argparse
from tqdm import tqdm
import os
from shutil import copyfile


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        type=str,
                        required=True,
                        help="Input path for abnormal data.")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="Output path for combined data.")
    parser.add_argument("--adversarial_successful_failed_file",
                        type=str,
                        required=False,
                        help="Index of samples for references in clean data")
    parser.add_argument("--change_to", type=str,required = True)
    parser.add_argument("--failed_flag", type=str,required = True)
    args = parser.parse_args()

    sents = []
    labels = []
    IDs = []
    if not args.adversarial_successful_failed_file:
        f = open(args.input_file, "r", encoding="utf-8-sig")
        data = csv.reader(f, delimiter=",")
        for (i, row) in enumerate(data):
            if i == 0 or row[4] != args.failed_flag:
                continue

            id = i
            sent = row[0]
            label = args.change_to
            sents.append(sent)
            labels.append(label)
            IDs.append(id)
        f.close()
    else:
        data_clean = pandas.read_csv(args.input_file, encoding="utf-8-sig",index_col=0)
        data_clean_0 = data_clean[data_clean["1"] != 0].values

        f_adversarial = pandas.read_csv(args.adversarial_successful_failed_file,sep="\t")
        reference_index = f_adversarial['Unnamed: 0'].values

        for (j, data) in enumerate(data_clean_0):
            if (j + 1) not in reference_index:
                continue

            id = j + 1
            sent = data[0]
            label = args.change_to
            sents.append(sent)
            labels.append(label)
            IDs.append(id)

    dataframe = pandas.DataFrame({'': IDs, 'sentence': sents, 'label': labels})
    dataframe.to_csv(args.output_file, sep='\t', index=False)

if __name__ == "__main__":
    main()