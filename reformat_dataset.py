import os
from shutil import copy

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# path to full, raw dataset
dataset_name = "full_siamese_dataset_extended_oldaeremoved"
raw_ds = "RAW_siamese_dataset_extended"
ds = os.path.join(os.getcwd(), "datasets", raw_ds)
new_pth = os.path.join(os.getcwd(), "datasets", dataset_name)

if not os.path.exists(new_pth):
    os.mkdir(new_pth)
    os.mkdir(os.path.join(new_pth, "train"))
    os.mkdir(os.path.join(new_pth, "test"))


# traverse root directory, and list directories as dirs and files as files
map_arr = []
t1 = {
    "train": [],
    "test": []
}
t2 = {
    "train": [],
    "test": []
}
t3 = {
    "train": [],
    "test": []
}
t4 = {
    "train": [],
    "test": []
}
ae = {
    "train": [],
    "test": []
}
current_id = 0

for tp in ["train", "test"]:
    datasets = os.path.join(ds, tp)
    print("Creating file mapping...")
    for root, dirs, files in tqdm(os.walk(datasets)):
        for file in files:
            label = file.split("_")[2]
            if label[0] in 'TAE':
                if label[0] == 'T' or "ART" in label:
                    if "ART" in label:
                        label = "AE"
                    name = label + "_" + str(current_id) + ".csv"
                    map_arr.append([name, root + os.sep + file])
                    if label[1] == "1":
                        t1[tp].append([name, root + os.sep + file])
                    elif label[1] == "2":
                        t2[tp].append([name, root + os.sep + file])
                    elif label[1] == "3":
                        t3[tp].append([name, root + os.sep + file])
                    elif label[1] == "4":
                        t4[tp].append([name, root + os.sep + file])
                    else:
                        ae[tp].append([name, root + os.sep + file])
                    current_id += 1

mapping = pd.DataFrame(data=map_arr, columns=["new_id", "RAW_path"])
mapping.to_csv(os.path.join(os.getcwd(), "datasets", dataset_name + "_to_" + raw_ds + ".csv"))

for tp in ["train", "test"]:
    X = t1[tp] + t2[tp] + t3[tp] + t4[tp] + ae[tp]

    new_path = os.path.join(new_pth, tp)
    print("Copying " + tp + " dataset...")
    for x in tqdm(X):
        name, root_pth = x
        copy(root_pth, os.path.join(new_path, name))
    
print("Finished")
