import argparse
import csv
import codecs
import glob
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--source", help="flag for path to data")
parser.add_argument("--target", help="flag for path to destination (must have a folder there)")
parser.add_argument("--include", help="flag for including files containing the given string")
# parser.add_argument("--exclude", help="flag for excluding files containing the given string")
parser.add_argument("--type", help="whether using Ines' new files or originals")
args = parser.parse_args()


def load(src):
    reader = csv.reader(codecs.open(src, 'rU', 'utf-8'))
    input_list = []
    for row in tqdm(reader):
        input_list.append(row)
    return input_list


def minimise_for_original(src):
    input_list = load(src)
    print(input_list[len(input_list) - 1])
    to_cancel = input("Cancel? ")
    if to_cancel.lower() == "yes":
        return
    response = input("Delete? ")
    if response.lower() == "yes":
        del input_list[len(input_list) - 1]

    for i, row in enumerate(input_list):
        entry = input_list[i][2:9]
        entry.append(input_list[i][-4])  # xMin, xMax, yMin, yMax, dye 1, dye 2, dye 3, neutral, cell area
        input_list[i] = entry

    print(src.split('/')[-1][:-4] + ".p")  # Remove '.csv' extension from src name

    print(len(input_list))
    print(input_list[0])

    file_name = args.target + src.split('/')[-1][:-4] + '.p'  # Remove '.csv' extension from src name

    p_file = open(file_name, 'wb')

    pickle.dump(input_list, p_file)

    p_file.close()


def minimise_for_ines(src):
    input_list = load(src)
    print(input_list[len(input_list) - 1])
    to_cancel = input("Cancel? ")
    if to_cancel.lower() == "yes":
        return
    response = input("Delete? ")
    if response.lower() == "yes":
        del input_list[len(input_list) - 1]

    for i, row in enumerate(input_list):
        input_list[i] = [input_list[i][1]] + input_list[i][3:]  # all cells (ignore), Dye 2 cells (cancer clusters), region Area (ignore), xMax, xMin, yMax, yMin

    file_name = args.target + src.split('/')[-1][:-4] + '.p'  # Remove '.csv' extension from src name

    p_file = open(file_name, 'wb')

    pickle.dump(input_list, p_file)

    p_file.close()


if __name__ == '__main__':
    for file in glob.glob(args.source + '*.csv'):
        if (args.include is None) or args.include in file:
            print("Minimising", file, "...")
            if args.type == "New":
                minimise_for_ines(file)
            else:
                minimise_for_original(file)
        # elif (args.exclude is None) or args.exclude not in file:
        #     print("Minimising", file, "...")
        #     if args.type == "New":
        #         minimise_for_ines(file)
        #     else:
        #         minimise_for_original(file)






# End of file
