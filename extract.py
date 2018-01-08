import csv
import codecs
import glob
import pickle
from tqdm import tqdm


def load(src):
    reader = csv.reader(codecs.open(src, 'rU', 'utf-8'))
    input_list = []
    for row in tqdm(reader):
        input_list.append(row)
    return input_list


def minimise(src):
    input_list = load(src)
    print(input_list[len(input_list) - 1])
    to_cancel = input("Cancel? ")
    if to_cancel.lower() == "yes":
        return
    response = input("Delete? ")
    if response.lower() == "yes":
        del input_list[len(input_list) - 1]

    for i, row in enumerate(input_list):
        input_list[i] = input_list[i][2:9]

    file_name = './' + src[:-4] + '.p'

    p_file = open(file_name, 'wb')

    pickle.dump(input_list, p_file)


if __name__ == '__main__':
    for file in glob.glob('*.csv'):
        print("Minimising", file, "...")
        minimise(file)
