import csv
import glob
import os
import codecs
import zipfile
from tqdm import tqdm


# Our dataset is very deep - only 61 examples, but ~100 GB in total. So we'll remove completely unnecessary fields
# to reduce the size to ~10 GB for better memory efficiency and convenience.

def load_and_strip(src):
    reader = csv.reader(codecs.open(src, 'rU', 'utf-8'))
    with open('./test/_' + src, 'w', newline='') as dest:
        writer = csv.writer(dest, delimiter=',')
        next(reader)
        for row in tqdm(reader):
            writer.writerow((row[1], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[15],
                            row[16], row[17], row[18], row[19], row[20], row[21], row[22], row[23], row[24], row[25]))
    os.remove(src)

if __name__ == '__main__':
    for file in glob.glob('*.zip'):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(file)

    for file in glob.glob('*.csv'):
        load_and_strip(file)
