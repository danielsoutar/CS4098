import argparse
import csv
import glob
import os
import codecs
import zipfile
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--source", help="flag for path to data")
parser.add_argument("--target", help="flag for path to destination (must have a folder there)")
parser.add_argument("--include", help="flag for including files containing the given string")
parser.add_argument("--type", help="flag for type of file (small/cancer bud file or large/cd3-cd8 cell file)")
args = parser.parse_args()


def load_and_strip(src):
    reader = csv.reader(codecs.open(src, 'rU', 'utf-8'))
    row = next(reader)
    if len(row) == 68 and args.type.lower() == "large":  # Massive files
        with open(args.target + 'stripped/full_dataset/' + src.split("/")[-1], 'w+', newline='') as dest:
            writer = csv.writer(dest, delimiter=',')
            for i, row in tqdm(enumerate(reader)):
                try:
                    writer.writerow((row[4], row[5], row[6], row[7], row[22], row[29]))  # xMin, xMax, yMin, yMax, dye 3 (cd3), dye 4 (cd8)
                except:
                    print("i:", i)
                    print("row:", row)
    elif len(row) == 107 and args.type.lower() == "small":  # Smaller files - tumour buds of varying sizes
        with open(args.target + 'stripped/full_dataset/' + src.split("/")[-1], 'w+', newline='') as dest:
            writer = csv.writer(dest, delimiter=',')
            for i, row in tqdm(enumerate(reader)):
                writer.writerow((row[4], row[6], row[-6], row[-4], row[-3], row[-2], row[-1]))  # all cells, Dye 2 cells (cancer clusters), region Area, xMax, xMin, yMax, yMin
    elif len(row) == 107 and args.type.lower() != "small":
        pass
    elif len(row) == 68 and args.type.lower() != "large":
        pass
    else:
        raise Exception("Unexpected input!")
    print(src.split("/")[-1], len(row))


if __name__ == '__main__':
    for file in glob.glob(args.source + '*.zip'):
        print("Unzipping", file, "...")
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(file)
        print(file, " unzipped ...")

    for file in glob.glob(args.source + '*.csv'):
        print("Stripping", file, "...")
        if (args.include is None) or args.include in file:
            load_and_strip(file)
    os.system('say "All files stripped in this batch."')


# End of file
