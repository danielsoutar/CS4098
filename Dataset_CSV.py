import csv
import glob
import pickle

csvfile = "./outcomes.csv"
reader = csv.reader(open(csvfile, "r"), delimiter=",")
outcomes = list(reader)


def find(code):
    for outcome in outcomes:
        if outcome[0] == code:
            return outcome[1], outcome[2]
    raise Exception("Error: code not in outcomes")


for name in glob.glob("./Ratio Heatmaps/*.p"):
    with open(name, "rb") as pickled_file:
        print(name.split("/")[-1][:-2])
        dataset = pickle.load(pickled_file)

        X, y, t, d, cluster_sizes = dataset["X"], dataset["y"], dataset["t"], dataset["d"], dataset["range"]

        entries = []

        for (i, lymph_name), (j, cluster_name) in zip(enumerate(sorted(glob.glob("./lymphocytes/*.csv"))),
                                                      enumerate(sorted(glob.glob("./LargerDataset/*.p")))):
            lymph_code = lymph_name.split('/')[-1][0:-4]

            code = lymph_name.split('/')[-1][0:-2]

            if "SN" in code:
                code = code[0:4]
            else:
                code = code.split("__")[1][0:4]

            acc = 0
            counter = 0

            (row, col) = X[i].shape

            for r in range(row):
                for c in range(col):
                    if X[i][r][c] != -1:
                        counter += 1
                        acc += X[i][r][c]

            my_mean = acc / counter

            entry = [code, str(my_mean)]

            entries.append(entry)

        for i, entry in enumerate(entries):
            surv, death = find(entry[0])
            entries[i].append(surv)
            entries[i].append(death)

        with open("./R_Means/" + name.split("/")[-1][:-2] + "_means" + ".csv", "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(("Case No.", "Mean", "DiseaseSpecificSurvival", "DiseaseSpecificDeath"))
            for entry in entries:
                writer.writerow((entry))


# # End of file
