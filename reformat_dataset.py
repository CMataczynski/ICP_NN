import os
from shutil import copy

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# path to full, raw dataset
dataset_name = "full_splitted_dataset"
datasets = os.path.join(os.getcwd(), "datasets", "RAW_splitted")
new_path = os.path.join(os.getcwd(), "datasets", dataset_name)
split = 0.25
# traverse root directory, and list directories as dirs and files as files
map_arr = []
t1 = []
t2 = []
t3 = []
t4 = []
ae = []
current_id = 0
print("Creating file mapping...")
for root, dirs, files in tqdm(os.walk(datasets)):
    for file in files:
        if file[0] in 'TAE':
            prefix = file.split("_")[0]
            name = prefix + "_" + str(current_id) + ".csv"
            map_arr.append([name, root + os.sep + file])
            if file[1] == "1":
                t1.append([name, root + os.sep + file])
            elif file[1] == "2":
                t2.append([name, root + os.sep + file])
            elif file[1] == "3":
                t3.append([name, root + os.sep + file])
            elif file[1] == "4":
                t4.append([name, root + os.sep + file])
            else:
                ae.append([name, root + os.sep + file])
            current_id += 1

mapping = pd.DataFrame(data=map_arr, columns=["new_id", "RAW_path"])
mapping.to_csv(os.path.join(os.getcwd(), "datasets", dataset_name + "_to_RAW_mapping.csv"), sep=';')

t1_train, t1_test = train_test_split(t1, test_size=split)
t2_train, t2_test = train_test_split(t2, test_size=split)
t3_train, t3_test = train_test_split(t3, test_size=split)
t4_train, t4_test = train_test_split(t4, test_size=split)
ae_train, ae_test = train_test_split(ae, test_size=split)

X_train = t1_train + t2_train + t3_train + t4_train + ae_train
X_test = t1_test + t2_test + t3_test + t4_test + ae_test

# X_train = X_train + X_test

if not os.path.exists(new_path):
    os.mkdir(new_path)
    os.mkdir(os.path.join(new_path, "train"))
    os.mkdir(os.path.join(new_path, "test"))

if not os.path.exists(os.path.join(new_path, "test")):
    os.mkdir(os.path.join(new_path, "test"))

if not os.path.exists(os.path.join(new_path, "train")):
    os.mkdir(os.path.join(new_path, "train"))

print("Copying training dataset...")
for x in tqdm(X_train):
    name, root_pth = x
    copy(root_pth, os.path.join(new_path, "train", name))
print("Copying testing dataset...")
for x in tqdm(X_test):
    name, root_pth = x
    copy(root_pth, os.path.join(new_path, "test", name))
print("Finished")
