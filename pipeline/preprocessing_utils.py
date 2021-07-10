import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def create_dataset_reference_file(base_dir='/content/drive/MyDrive/PlantVillage/color', output_file='/content/drive/MyDrive/PlantVillage/color/dataset_reference.csv', classes=['GLP', 'CR', 'NLB', 'H']):
    # Considering access to Google Drive
    files = []
    for target_class in os.listdir(base_dir):
        if target_class in classes:
            target_class_folder = base_dir+f'/{target_class}'
            for file_ in os.listdir(target_class_folder):
                if file_.endswith('.JPG') or file_.endswith('.jpg'):
                    files.append({
                        'filename': file_,
                        'location': target_class_folder+f'/{file_}',
                        'class': target_class
                    })

    dataset_reference = pd.DataFrame.from_records(files)

    # Save for later use
    dataset_reference.to_csv(output_file, index=False)


def train_test_val_split_dataset(dataset_reference, train_size, test_size, val_size=0.05):
    X_train, X_test, y_train, y_test = train_test_split(dataset_reference.drop(
        columns='class'), dataset_reference['class'], train_size=train_size, test_size=test_size, random_state=2, shuffle=True)

    X_train['class'] = y_train
    X_test['class'] = y_test

    train_df = X_train.copy()

    X_train, X_val, y_train, y_val = train_test_split(train_df.drop(
        columns='class'), train_df['class'], train_size=(1-val_size), test_size=val_size, random_state=2, shuffle=True)

    X_train['class'] = y_train
    X_val['class'] = y_val

    train_df = X_train.copy()
    test_df = X_test.copy()
    val_df = X_val.copy()

    return train_df, test_df, val_df


def open_img(img_location):
    # Open image and load into memory
    img_PIL = Image.open(img_location)
    return img_PIL


def resize_img(img):
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized)
    return img_array


def explained_variance(df, chart=False):
    pca = PCA(whiten=True)
    pca.fit(df)

    # Getting the cumulative variance
    var_cumu = np.cumsum(pca.explained_variance_ratio_)*100

    # How many PCs explain 95% of the variance?
    k = np.argmax(var_cumu > 95)
    print("Number of components explaining 95% variance: " + str(k))
    # print("\n")
    if chart:
        plt.figure(figsize=[10, 5])
        plt.title('Cumulative Explained Variance explained by the components (RED)')
        plt.ylabel('Cumulative Explained variance')
        plt.xlabel('Principal components')
        plt.axvline(x=k, color="k", linestyle="--")
        plt.axhline(y=95, color="r", linestyle="--")
        ax = plt.plot(var_cumu)
    return k


def compress_image(img, pca_rgb):
    inverted_lst = []
    for idx, pca in enumerate(pca_rgb):
        channel = img[:, :, idx]
        channel_transformed = pca.transform(channel.flatten().reshape(1, -1))
        channel_inverted = pca.inverse_transform(channel_transformed)
        inverted_lst.append(channel_inverted.reshape(64, 64).astype(np.uint8))
    return np.dstack(tuple(inverted_lst))


def apply_pca(df, img_lst, pca_red, pca_green, pca_blue, save_dir):
    # Use PCA to compress all images and save into the preprocessing folders
    try:
        df = df.drop(columns='preprocessing_location')
    except KeyError:
        df = df
    cnt = 0
    for idx, row in df.iterrows():
        save_loc = save_dir + f'/image_{idx}.jpg'
        arr = compress_image(img_lst[idx], [pca_red, pca_green, pca_blue])
        img = Image.fromarray(arr, 'RGB')
        img.save(save_loc)
        df.loc[idx, 'preprocessing_location'] = save_loc
        cnt += 1

    return df


def split_channels(img):
    return img[:,:,0],img[:,:,1],img[:,:,2]

def combine(img_list):
    r,g,b=[],[],[]
    for img in img_list:
        red, green, blue = split_channels(img)
        r.append(pd.DataFrame(red.flatten().reshape(1,-1)))
        g.append(pd.DataFrame(green.flatten().reshape(1,-1)))
        b.append(pd.DataFrame(blue.flatten().reshape(1,-1)))

    return pd.concat(r).reset_index(drop=True),pd.concat(g).reset_index(drop=True),pd.concat(b).reset_index(drop=True)