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
parser.add_argument("--type", help="whether using new cell files, cluster files, or lymphocyte files")
args = parser.parse_args()


def load(src):
    reader = csv.reader(codecs.open(src, "rU", "utf-8"))
    input_list = []
    for row in tqdm(reader):
        input_list.append(row)
    return input_list


def minimise_new_cell_file(src):
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

    file_name = args.target + src.split('/')[-1][:-4] + ".p"  # Remove ".csv" extension from src name

    p_file = open(file_name, "wb")

    pickle.dump(input_list, p_file)

    p_file.close()


def minimise_new_cluster_file(src):
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

    file_name = args.target + src.split('/')[-1][:-4] + ".p"  # Remove ".csv" extension from src name

    p_file = open(file_name, "wb")

    pickle.dump(input_list, p_file)

    p_file.close()


# Annoyingly, the actual sizes of these resulting files in the aggregate lead to 23GB of space. Not ideal, and weirdly less space efficient. I ended up not
# using this function.
def minimise_new_lymphocyte_file(src):
    input_list = load(src)
    print(input_list[len(input_list) - 1])

    to_cancel = input("Cancel? ")
    if to_cancel.lower() == "yes":
        return
    response = input("Delete? ")
    if response.lower() == "yes":
        del input_list[len(input_list) - 1]

    for i, row in enumerate(input_list):
        input_list[i] = input_list[i][:]  # xMin, xMax, yMin, yMax, dye 3 (cd3), dye 4 (cd8)

    file_name = args.target + src.split('/')[-1][:-4] + ".p"  # Remove ".csv" extension from src name

    p_file = open(file_name, "wb")

    pickle.dump(input_list, p_file)

    p_file.close()


def minimise_validation_set_tb(src):
    """
    This file contains all of the patients, unfortunately.
    Need to split up and create multiple output files.
    Need to get name (line[0]), number of dye2 positive cells (line[6]),
    and the Xmax/Xmin/Ymax/Ymin (line[-4:]).
    """
    input_list = load(src)

    # Validate the number of distinct files here
    file_data = {}

    for i, row in enumerate(input_list):
        if i == 0:
            continue

        entry = [row[54], row[5], row[4], row[7], row[6]]  # dye2+ cells, xmax, xmin, ymax, ymin
        filename = row[0]

        if filename in file_data:
            file_data[filename].append(entry)
        else:
            file_data[filename] = [entry]

    print("Number of files:", len(file_data.keys()))

    for filename in file_data.keys():
        output_filename = "/Volumes/Dan Media/ValidationDataset/TB_pickle_files/" + filename.split('\\')[-1][0:-4] + ".p"  # get the last token with the patient code
        print(output_filename)
        p_file = open(output_filename, "wb")
        pickle.dump(file_data[filename], p_file)
        p_file.close()


def clear_data(file_data):
    print("in clear_data()")
    for key in file_data.keys():
        code = key.split('\\')[-1]  # get the last token with the patient code
        output_filename = "/Volumes/Dan Media/ValidationDataset/lymphocyte_csv_files/" + code + ".csv"
        with open(output_filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(file_data[key])


def minimise_validation_set_lymphocyte(src):
    """
    This file contains all of the paients. This means a mammoth 100+GB of data
    per file. Have to stream through the file rather than store in intermediate list,
    because there is LITERALLY not enough space to do otherwise.
    Need to split up and create multiple output files.
    Need to get name (line[0]), number of dye 3/4 for CD3/CD8 (respectively, line[22]/line[29]),
    and Xmax/Xmin/Ymax/Ymin (line[4:8]).
    """

    # Validate the number of distinct files here
    file_data = {}

    threshold = 2000000
    unique_keys = set()

    reader = csv.reader(codecs.open(src, "rU", "utf-8"))
    for i, row in tqdm(enumerate(reader)):
        # Skip the first row.
        if i == 0:
            continue

        if i % threshold == 0:
            clear_data(file_data)
            file_data = {}

        entry = [row[4], row[5], row[6], row[7], row[22], row[29]]  # xmin, xmax, ymin, ymax, dye3+, dye4+ cells
        filename = row[0]
        if filename not in unique_keys:
            unique_keys.add(filename)

        if filename in file_data:
            file_data[filename].append(entry)
        else:
            file_data[filename] = [entry]

    print("Number of files:", len(unique_keys))

    clear_data(file_data)

    return unique_keys


def get_all_validation_cohort_codes():
    all_codes = load("/Users/soutar/Documents/Computer Science/CS4098/Patient_codes_validation_cohorts.csv")
    all_codes_japan = set()
    all_codes_edinburgh = set()
    for (SN_code, HU_code) in all_codes[1:]:
        ecode, jcode = SN_code, HU_code
        while len(ecode) != 3:
            ecode = "0" + ecode
        while len(jcode) != 3:
            jcode = "0" + jcode
        ecode, jcode = "SN" + ecode, "HU" + jcode
        if ecode != "SN000":
            all_codes_edinburgh.add(ecode)
        if jcode != "SN000":
            all_codes_japan.add(jcode)

    return {"EDINBURGH CODES": all_codes_edinburgh, "JAPANESE CODES": all_codes_japan}

if __name__ == "__main__":
    for file in glob.glob(args.source + "*.csv"):
        if (args.include is None) or args.include in file:
            print("Minimising", file, "...")
            if args.type.lower() == "clusters":
                minimise_new_cluster_file(file)
            elif args.type.lower() == "lymphocytes":
                minimise_new_lymphocyte_file(file)
            elif args.type.lower() == "cancer cells":
                minimise_new_cell_file(file)




# End of file
