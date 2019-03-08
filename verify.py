# verify.py: a module for verifying various properties and conditions of our data.


import codecs
import csv
import extract
import glob
import os
import pickle
from tqdm import tqdm


def verify_codes_match_in_directory(src_file, code_extract_func, target_directory, target_code_extract_func):
    """
    Given a source file that we want to split, verify that the (unique) codes within it
    match precisely the names of the files in the target directory.

    code_extract_func takes a row and returns a patient code from it.
    target_code_extract_func takes a filename and returns a patient code from it.

    This assumes the file is massive - i.e. cannot fit into memory.
    """
    reader = csv.reader(codecs.open(src_file, "rU", "utf-8"))

    src_file_codes = set()

    for i, row in tqdm(enumerate(reader)):
        if i == 0:
            continue

        code = code_extract_func(row)
        if code not in src_file_codes:
            src_file_codes.add(code)

    target_dir_codes = set()

    for i, target_file_name in tqdm(enumerate(sorted(glob.glob(target_directory)))):
        code = target_code_extract_func(target_file_name)
        if code not in target_dir_codes:
            target_dir_codes.add(code)

    assert(src_file_codes == target_dir_codes)


def verify_codes_match_full_list(src_directory, code_extract_func, codes_list_file):
    """
    Given a src directory containing files with names containing patient codes, verify that
    these codes match a reference list of all the patient codes.

    code_extract_func takes a row and returns a patient code from it.

    Assumes the reference list file is a pickle file
    """
    reference_codes = set(pickle.load(open("reference_code_list.p", "rb")))

    src_dir_codes = set()

    for i, src_file_name in tqdm(enumerate(sorted(glob.glob(src_directory)))):
        code = code_extract_func(src_file_name)
        if code not in src_dir_codes:
            src_dir_codes.add(code)

    assert(src_dir_codes == reference_codes)


def verify_sizes(reference_size, container):
    """
    Given a reference size, verify that the container has the specified number of elements.

    Transform into a list to ensure that different containers are handled correctly.
    """
    assert len(list(container)) == reference_size


def verify_num_files(reference_size, reference_extension, directory):
    """
    Given a reference size and extension, verify that the directory has the specified no
    of files (with all having same file extension). Note the directory should have the
    pattern '/path/to/dir/*'.

    Iterate over the files to ensure we get their file extension.
    """

    if directory[-1] != "*":
        raise Exception("Need to include wildcard character")

    count = 0

    for i, filename in enumerate(sorted(glob.glob(directory))):
        if reference_extension not in filename:
            raise Exception("Directory should have files of same type")
        count = i  # Indexes from 0, so add 1 at the end

    count += 1

    return count, (count == reference_size)


def num_objects_per_file(obj_type, src_directory, open_func, count_func, code_extract_func, target_directory, dry_run=True):
    """
    Given a source directory and a way to access data in each file in that directory,
    get the count of objects for each file using count_func() and write them to the
    the metadata store, named using the code extracted by code_extract_func.

    The output should contain the type of the object checked and the actual value.
    """
    for i, filename in enumerate(sorted(glob.glob(src_directory))):
        data = open_func(filename)
        num_objs = count_func(data)
        output_filename = code_extract_func(filename) + ".txt"

        print(target_directory + output_filename + " --> " + str(num_objs))

        output_filename = target_directory + output_filename

        current_content = []

        with open(output_filename, 'r') as f:
            current_content = f.readlines()

        clu_in_content = False
        tb_in_content = False

        for i in range(len(current_content)):
            if "CLU" in current_content[i] and obj_type == "CLU":
                clu_in_content = True
                current_content[i] = "Num objects for " + obj_type + ": " + str(num_objs) + "\n"

            if "TB" in current_content[i] and obj_type == "TB":
                tb_in_content = True
                current_content[i] = "Num objects for " + obj_type + ": " + str(num_objs) + "\n"

        if (not clu_in_content and obj_type == "CLU") or (not tb_in_content and obj_type == "TB"):
            obj_line = "Num objects for " + obj_type + ": " + str(num_objs) + "\n"
            current_content.append(obj_line)
            print()
            print(output_filename + " has been modified")
            print()

        print(current_content)

        if not dry_run:
            if not os.path.exists(output_filename):
                open(output_filename, 'a').close()

            with open(output_filename, 'w') as f:
                for line in current_content:
                    f.write(line)


def get_patients_from_job_lines_directory(src_directory, code_extract_func):
    """
    This directory contains all the Job lines for a given cohort.
    Given a source directory and a code_extract_func to extract the patient codes
    from the files in that directory, return the codes.
    """
    codes = set()
    for i, filename in enumerate(sorted(glob.glob(src_directory))):
        code = code_extract_func(filename)
        if code not in codes:
            codes.add(code)
        else:
            print("WARNING: duplicate code in src_directory:", code)

    return codes


def get_patients_in_csv_file(src_directory, code_extract_func):
    """
    This .csv file contains all the TB data for a cohort.
    Given a source directory and a code_extract_func to extract the patient codes
    from the files in that directory, return the codes.

    The expected number of unique codes should equal 56/62 for Edinburgh/Japan respectively.
    """
    input_list = extract.load(src_directory)

    # Validate the number of distinct files here
    filenames = set()

    for i, row in enumerate(input_list):
        if i == 0:
            continue
        filename = code_extract_func(row[0])

        if filename not in filenames:
            print(filename)
            filenames.add(filename)

    print("Number of files:", len(filenames))

    return filenames


def get_patients_in_massive_csv_file(src_directory, code_extract_func):
    """
    This src_directory contains all the LYM data for a cohort.
    Given a source directory and a code_extract_func to extract the patient codes
    from the files in that directory, return the codes.

    The expected number of unique codes should equal 56/62 for Edinburgh/Japan respectively.
    """

    # Validate the number of distinct files here
    filenames = set()

    reader = csv.reader(codecs.open(src_directory, "rU", "utf-8"))
    for i, row in tqdm(enumerate(reader)):
        # Skip the first row.
        if i == 0:
            continue

        filename = code_extract_func(row[0])

        if filename not in filenames:
            print(filename)
            filenames.add(filename)

    print("Number of files:", len(filenames))

    return filenames


if __name__ == "__main__":
    ### VERIFY NUMBER OF OBJECTS AND FILES FOR VALIDATION COHORTS
    # EDINBURGH TB (X)
    # assert(len(get_patients_in_csv_file("/Users/soutar/Desktop/Daniel_validation_set_data_TB/Edinburgh Cohort/TB Edinburgh.csv", lambda name: name.split('\\')[-1][0:-4])) == 54)
    # assert(len(get_patients_in_csv_file("/Users/soutar/Desktop/Daniel_validation_set_data_TB/Edinburgh Cohort/TB 259.csv", lambda name: name.split('\\')[-1][0:-4])) == 1)
    # assert(len(get_patients_in_csv_file("/Users/soutar/Desktop/Daniel_validation_set_data_TB/Edinburgh Cohort/TB on SN222.csv", lambda name: name.split('\\')[-1][0:-4])) == 1)

    # JAPAN TB ()
    # filenames = get_patients_in_csv_file("/Volumes/Dan Media/ValidationDataset/TB_pickle_files/Japanese_Cohort/TB data Japanese cases.csv", lambda name: name[0:-4])
    # assert(len(filenames) != 62)  # <-- missing HU596

    # all_codes_dict = extract.get_all_validation_cohort_codes()
    # all_codes_edinburgh, all_codes_japan = all_codes_dict["EDINBURGH CODES"], all_codes_dict["JAPANESE CODES"]

    # EDINBURGH LYM (X)
    # filenames = get_patients_in_massive_csv_file("/Volumes/Elements/Daniel validation set data/Halo archive 2018-12-20 19-11/object_results.csv", lambda name: name.split('\\')[-1][0:-4])
    # print(filenames)
    # print(len(filenames))
    # print(len(filenames) == 56)

    # print(all_codes_edinburgh.symmetric_difference(filenames))

    # # JAPANESE LYM (X)
    # filenames = get_patients_in_massive_csv_file("/Volumes/Dan Media/ValidationDataset/failed_lymphocyte_pickle_files/*.p", lambda name: name[0:-2])
    # print(filenames)
    # print(len(filenames))
    # print(len(filenames) == 62)

    # print(all_codes_japan.symmetric_difference(filenames))

    # EDINBURGH LINES ()
    # filenames = get_patients_from_job_lines_directory("/Users/soutar/Desktop/Daniel_validation_set_data_TB/Edinburgh_Cohort/Line/JOB_LINES/*", lambda name: name.split("/")[-1].split("_")[0])
    # print(len(filenames) == 56)

    # print(all_codes_edinburgh.symmetric_difference(filenames))

    # JAPANESE LINES (X)
    # filenames = get_patients_from_job_lines_directory("/Users/soutar/Desktop/Daniel_validation_set_data_TB/Japanese_Cohort/Line/JOB_LINES/*", lambda name: name.split("/")[-1].split("_")[0])
    # print(len(filenames) == 62)

    # print(all_codes_japan.symmetric_difference(filenames))

    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    ######## META DATA COLLECTION
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################

    # # ORIGINAL DATASET: TB
    tb_src_directory = "/Volumes/Dan Media/OriginalDataset/TB_pickle_files/*"

    # open_func_tb = lambda filename: pickle.load(open(filename, "rb"))
    # count_func_clusters = lambda data: sum([1 for x in data if int(x[0]) > 0])
    # count_func_tb = lambda data: sum([1 for x in data if int(x[0]) > 0 and int(x[0]) < 5])
    # code_extract_func_tb = lambda filename: filename.split('/')[-1][0:4]

    # num_objects_per_file("CLU", tb_src_directory, open_func_tb, count_func_clusters, code_extract_func_tb, "/Users/soutar/Documents/Computer Science/CS4098/METADATA/ORIGINAL_DATA/", dry_run=False)
    # num_objects_per_file("TB", tb_src_directory, open_func_tb, count_func_tb, code_extract_func_tb, "/Users/soutar/Documents/Computer Science/CS4098/METADATA/ORIGINAL_DATA/", dry_run=True)

    # ORIGINAL DATASET: LYM
    # lym_src_directory = "/Volumes/Dan Media/OriginalDataset/lymphocyte_csv_files/*"

    # open_func_lym = lambda filename: extract.load(filename)
    # count_func_lym = lambda data: sum([1 for x in data if int(x[-2]) > 0 or int(x[-1]) > 0])
    # count_func_cd3 = lambda data: sum([1 for x in data if int(x[-2]) > 0])
    # count_func_cd8 = lambda data: sum([1 for x in data if int(x[-1]) > 0])
    # code_extract_func_lym = lambda filename: filename.split('/')[-1][0:4]

    # num_objects_per_file("CD3&CD8", lym_src_directory, open_func_lym, count_func_lym, code_extract_func_lym, "/Users/soutar/Documents/Computer Science/CS4098/METADATA/ORIGINAL_DATA/", dry_run=True)
    # num_objects_per_file("CD3", lym_src_directory, open_func_lym, count_func_cd3, code_extract_func_lym, "/Users/soutar/Documents/Computer Science/CS4098/METADATA/ORIGINAL_DATA/", dry_run=True)
    # num_objects_per_file("CD8", lym_src_directory, open_func_lym, count_func_cd8, code_extract_func_lym, "/Users/soutar/Documents/Computer Science/CS4098/METADATA/ORIGINAL_DATA/", dry_run=True)

    #

    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################
    # VALIDATION DATASET: TB
    # tb_src_directory = "/Volumes/Dan Media/ValidationDataset/TB_pickle_files/*"

    # open_func_tb = lambda filename: pickle.load(open(filename, "rb"))
    # count_func_clusters = lambda data: sum([1 for x in data if int(x[0]) > 0])
    # count_func_tb = lambda data: sum([1 for x in data if int(x[0]) > 0 and int(x[0]) < 5])
    # code_extract_func_tb = lambda filename: filename.split('/')[-1][0:-2]

    # num_objects_per_file("CLU", tb_src_directory, open_func_tb, count_func_clusters, code_extract_func_tb, "/Users/soutar/Documents/Computer Science/CS4098/METADATA/VALIDATION_DATA/", dry_run=False)
    # num_objects_per_file("TB", tb_src_directory, open_func_tb, count_func_tb, code_extract_func_tb, "/Users/soutar/Documents/Computer Science/CS4098/METADATA/VALIDATION_DATA/", dry_run=False)

    # VALIDATION DATASET: LYM
    # lym_src_directory = "/Volumes/Dan Media/ValidationDataset/lymphocyte_csv_files/*"

    # open_func_lym = lambda filename: extract.load(filename)
    # count_func_lym = lambda data: sum([1 for x in data if int(x[-2]) > 0 or int(x[-1]) > 0])
    # count_func_cd3 = lambda data: sum([1 for x in data if int(x[-2]) > 0])
    # count_func_cd8 = lambda data: sum([1 for x in data if int(x[-1]) > 0])
    # code_extract_func_lym = lambda filename: filename[0:-4]

    # num_objects_per_file("CD3&CD8", lym_src_directory, open_func_lym, count_func_lym, code_extract_func_lym, "/Users/soutar/Documents/Computer Science/CS4098/METADATA/VALIDATION_DATA/", dry_run=False)
    # num_objects_per_file("CD3", lym_src_directory, open_func_lym, count_func_cd3, code_extract_func_lym, "/Users/soutar/Documents/Computer Science/CS4098/METADATA/VALIDATION_DATA/", dry_run=False)
    # num_objects_per_file("CD8", lym_src_directory, open_func_lym, count_func_cd8, code_extract_func_lym, "/Users/soutar/Documents/Computer Science/CS4098/METADATA/VALIDATION_DATA/", dry_run=False)
    ##############################################################################################################
    ##############################################################################################################
    ##############################################################################################################


# End of file
