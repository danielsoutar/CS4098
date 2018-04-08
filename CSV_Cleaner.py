import csv


code_mapping_file = "./10.5yr follow up survival EDI.csv"
reader_for_codes = csv.reader(open(code_mapping_file, "r"), delimiter=",")
code_entries = list(reader_for_codes)[1:]

code_mappings = {}

for i, entry in enumerate(code_entries):
    sn_code = entry[0].split("(")[0]
    key = sn_code
    my_code = entry[0].split("(")[-1].replace(")", "")
    if len(my_code) == 3:
        my_code = "0" + my_code

    code_mappings[sn_code] = my_code

print(code_mappings)

clinical_data_file = "./Grouped Clinical Data.csv"
reader_for_data = csv.reader(open(clinical_data_file, "r"), delimiter=",")

data_entries = list(reader_for_data)[1:]

rows = []

for i, entry in enumerate(data_entries):
    sn_code = entry[0]
    if sn_code in code_mappings.keys():
        row_code = code_mappings[sn_code]
        age, sex, pT, site, tum_type, diff = entry[1], entry[2], entry[3], entry[4], entry[8], entry[9]
        row = [row_code, age, sex, pT, site, tum_type, diff]
        rows.append(row)

rows = sorted(rows)

for row in rows:
    print(row)

cleaned_clinical_data_file = "./Cleaned Medical Data (exclude all NA).csv"

with open(cleaned_clinical_data_file, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for row in rows:
        writer.writerow((row))



































# End of file
