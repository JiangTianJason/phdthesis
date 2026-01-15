import argparse
import csv
import pandas
from tqdm import tqdm
import os,random
from shutil import copyfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abnormal_file",
                        type=str,
                        required=True,
                        help="Input path for abnormal data.")
    parser.add_argument("--abnormal_file_type",
                        type=str,
                        default='txt',
                        help="tsv or txt.")
    parser.add_argument("--abnormal_number",
                        type=int,
                        default=-1,
                        help="How many abnormal examples to extract.")
    parser.add_argument("--normal_file",
                        type=str,
                        required=True,
                        help="Input path for normal data.")
    parser.add_argument("--normal_file_type",
                        type=str,
                        default='tsv',
                        help="tsv or txt.")
    parser.add_argument("--normal_number",
                        type=int,
                        default=-1,
                        help="How many normal examples to extract.")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="Output path for combined data.")
    parser.add_argument("--output_file_type",
                        type=str,
                        default='tsv',
                        help="tsv or txt.")
    args = parser.parse_args()

    IDs = []
    sents = []
    labels = []
    i = 0

    # read in abnormal file
    if args.abnormal_file_type == 'tsv':
        with open(args.abnormal_file, "r", encoding="utf-8-sig") as f:
            data = csv.reader(f, delimiter=",")
            for (k,row) in enumerate(data):
                if row[0] == '0' or row[4] == "Failed":
                    continue
                sents.append(row[0])
                labels.append('1')
                i += 1
                IDs.append(k)
                if i >= args.abnormal_number and args.abnormal_number != -1:
                    break

    reference_index = IDs

    # read in normal file
    if args.normal_file_type == 'tsv':
        with open(args.normal_file, "r", encoding="utf-8-sig") as f:
            data = csv.reader(f, delimiter=",")
            for (j,row) in enumerate(data):
                if row[0] == 'Unnamed: 0' or (j not in reference_index):
                    continue
                sents.append(row[1])
                labels.append('0')
                i += 1
                IDs.append(j)
                if i >= args.normal_number + args.abnormal_number and args.normal_number != -1:
                    break

    # write to output file
    if args.output_file_type == 'tsv':
        dataframe = pandas.DataFrame({'': IDs, 'sentence': sents, 'label': labels})
        dataframe.to_csv('./temp/temp.tsv', sep='\t', index=False)
        print("output file in form of tsv built")
        out = open(args.output_file, 'w', encoding='utf-8')
        lines = []
        with open('temp/temp.tsv', 'r', encoding='utf-8') as infile:
            for (i, line) in enumerate(infile):
                if i == 0:
                    out.write(line)
                    continue
                lines.append(line)
            # random.shuffle(lines)
            # random.shuffle(lines)
            # random.shuffle(lines)
            for line in lines:
                out.write(line)
            out.close()

if __name__ == "__main__":
    main()