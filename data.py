import os
import pandas as pd
import numpy as np
import shutil
import cv2

dir = "/ext/Data/Dog_Breed_Identification/Kaggle"

dog_imgs_list_csv = os.path.join(dir, "labels.csv")
df = pd.read_csv(dog_imgs_list_csv)

for index,row in df.iterrows():
    breed = row["breed"]
    id = row["id"]
    img_file = id + ".jpg"
    breed_dir = os.path.join(dir, "train", breed)
    if not os.path.exists(breed_dir):
        os.mkdir(breed_dir)

    old_img_file = os.path.join(dir, "train0", img_file)
    shutil.copy(old_img_file, os.path.join(breed_dir, img_file))
    img = cv2.imread(old_img_file)
    flip_img = cv2.flip(img, 1)
    cv2.imwrite(os.path.join(breed_dir, "_"+img_file), flip_img)

print("train data done")

