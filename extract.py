import csv
import glob
import os
import codecs
import zipfile
from tqdm import tqdm


def load_and_strip(src):
    reader = csv.reader(codecs.open(src, 'rU', 'utf-8'))
    input_list = []
    for row in tqdm(reader):
        input_list.append(row)
    return input_list
