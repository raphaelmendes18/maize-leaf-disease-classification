__author__ = 'Raphael Mendes'
__date__ = '06/22/21'

from sklearn.decomposition import PCA
from preprocessing.utils import explained_variance, combine, resize_img, open_img, apply_pca, create_dataset_reference_file, train_test_val_split_dataset
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--BASE_DIR', type=str)
parser.add_argument('--OUTPUT_FILE', type=str)
parser.add_argument('--TRAIN_SIZE', type=float)
parser.add_argument('--TEST_SIZE', type=float)
parser.add_argument('--EXP_NAME', type=str)
parser.add_argument('--PREPROCESS_BASE_DIR', type=str)

# PARAMETERS
# BASE_DIR = '/content/drive/MyDrive/PlantVillage/color'
# OUTPUT_FILE = '/content/drive/MyDrive/PlantVillage/color/dataset_reference.csv'
# TRAIN_SIZE = 0.5
# TEST_SIZE = 0.5
# EXP_NAME = 'exp_1'
# PREPROCESS_BASE_DIR = f'/content/drive/MyDrive/PlantVillage/preprocessing/PCA/{EXP_NAME}'

args = parser.parse_args()
BASE_DIR = args.BASE_DIR
OUTPUT_FILE = args.OUTPUT_FILE
TRAIN_SIZE = args.TRAIN_SIZE
TEST_SIZE = args.TEST_SIZE
EXP_NAME = args.EXP_NAME
PREPROCESS_BASE_DIR = args.PREPROCESS_BASE_DIR

create_dataset_reference_file(base_dir=BASE_DIR, output_file=OUTPUT_FILE)
dataset_reference = pd.read_csv(OUTPUT_FILE)

train_df, test_df, val_df = train_test_val_split_dataset(
    dataset_reference, train_size=TRAIN_SIZE, test_size=TEST_SIZE)

# Open all images, resize and store in a list
img_lst = train_df['location'].apply(lambda loc: resize_img(open_img(loc)))

X_r, X_g, X_b = combine(img_lst)

# initialize PCA
pca_red = PCA(explained_variance(X_r), whiten=True)
pca_green = PCA(explained_variance(X_g), whiten=True)
pca_blue = PCA(explained_variance(X_g), whiten=True)

# fit PCA
pca_red.fit(X_r)
pca_green.fit(X_g)
pca_blue.fit(X_b)
print("\n")

# apply pca in train, val and test sets, this creates a new column named 'preprocessing_location'
train_df = apply_pca(train_df, img_lst, pca_red, pca_green,
                     pca_blue, save_dir=f'{PREPROCESS_BASE_DIR}/train')
test_df = apply_pca(test_df, img_lst, pca_red, pca_green,
                    pca_blue, save_dir=f'{PREPROCESS_BASE_DIR}/test')
val_df = apply_pca(val_df, img_lst, pca_red, pca_green,
                   pca_blue, save_dir=f'{PREPROCESS_BASE_DIR}/val')

# save the dataframes
train_df.to_csv(f'{PREPROCESS_BASE_DIR}/train.csv')
test_df.to_csv(f'{PREPROCESS_BASE_DIR}/test.csv')
val_df.to_csv(f'{PREPROCESS_BASE_DIR}/val.csv')
