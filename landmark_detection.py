import os
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

# Configuration
CSV_PATH = "train.csv"
BASE_PATH = "./images/"
SAMPLES = 20000
BATCH_SIZE = 16
EPOCHS = 15
IMG_SIZE = (224, 224)

def load_data(csv_path, samples):
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Returning empty dataframe.")
        return pd.DataFrame(columns=["id", "landmark_id"])
    
    df = pd.read_csv(csv_path)
    # Ensure we only use 'id' and 'landmark_id'
    if 'id' in df.columns and 'landmark_id' in df.columns:
        df = df[['id', 'landmark_id']]
    elif len(df.columns) >= 2:
        # Fallback if the columns are named differently (e.g. index 0 is id, index 1 or 2 is label)
        cols = df.columns.tolist()
        df = df[[cols[0], cols[-1]]]
        df.columns = ['id', 'landmark_id']
    
    df = df.head(samples)
    return df

def perform_eda(df):
    if df.empty:
        return
        
    num_classes = len(df["landmark_id"].unique())
    print("Size of data:", df.shape)
    print("Number of unique classes:", num_classes)

    data = pd.DataFrame(df['landmark_id'].value_counts()).reset_index()
    data.columns = ['landmark_id', 'count']
    
    print("\nTop 10 classes by frequency:")
    print(data.head(10))
    
    print("\nClasses with few datapoints:")
    print("<= 5 datapoints:", (data['count'].between(0, 5)).sum())
    print("6 to 10 datapoints:", (data['count'].between(6, 10)).sum())

    plt.figure(figsize=(10, 5))
    plt.hist(data['count'], bins=100, range=(0, data['count'].max() or 1000))
    plt.xlabel("Amount of images")
    plt.ylabel("Occurrences")
    plt.title("Distribution of Images per Class")
    plt.savefig("class_distribution.png")
    print("\nSaved class distribution histogram to class_distribution.png")
    plt.close()

def get_image_path(fname, base_path):
    fname_str = str(fname)
    # GLD datasets typically use the first 3 characters of the ID for folders
    if len(fname_str) >= 3:
        f1, f2, f3 = fname_str[0], fname_str[1], fname_str[2]
        return os.path.join(base_path, f1, f2, f3, f"{fname_str}.jpg")
    return os.path.join(base_path, f"{fname_str}.jpg")

def get_image_and_label(num, df, base_path):
    row = df.iloc[num]
    fname = row['id']
    label = row['landmark_id']
    
    img_path = get_image_path(fname, base_path)
    if os.path.exists(img_path):
        im = cv2.imread(img_path)
        if im is not None:
            return im, label
            
    # Return a black image if not found so the pipeline doesn't crash during missing images
    return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8), label

def get_batch(dataframe, start, batch_size, base_path, lencoder):
    image_array = []
    label_array = []
    end_img = min(start + batch_size, len(dataframe))
    
    for idx in range(start, end_img):
        im, label = get_image_and_label(idx, dataframe, base_path)
        im = cv2.resize(im, IMG_SIZE) / 255.0
        image_array.append(im)
        label_array.append(label)
        
    label_array = lencoder.transform(label_array)
    return np.array(image_array), np.array(label_array)

def build_model(num_classes):
    source_model = VGG19(weights=None, include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    model = Sequential()
    for layer in source_model.layers:
        model.add(layer)
        
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    
    optim = RMSprop(learning_rate=0.0001, momentum=0.09)
    model.compile(optimizer=optim,
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
    return model

def main():
    print("Loading data...")
    df = load_data(CSV_PATH, SAMPLES)
    if df.empty:
        print("Dataframe is empty. Ensure 'train.csv' is present. Exiting.")
        return
        
    perform_eda(df)
    
    lencoder = LabelEncoder()
    lencoder.fit(df["landmark_id"])
    num_classes = len(lencoder.classes_)
    
    print("\nBuilding model...")
    model = build_model(num_classes)
    model.summary()
    
    # Split train data up into 80% and 20% validation
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    print(f"\nTraining on: {len(train_df)} samples")
    print(f"Validation on: {len(val_df)} samples")
    
    for e in range(EPOCHS):
        print(f"\nEpoch: {e+1}/{EPOCHS}")
        train_df = train_df.sample(frac=1) # Shuffle data at the beginning of each epoch
        
        # Training
        batch_count = int(np.ceil(len(train_df) / BATCH_SIZE))
        for it in range(batch_count):
            X_train, y_train = get_batch(train_df, it * BATCH_SIZE, BATCH_SIZE, BASE_PATH, lencoder)
            if len(X_train) > 0:
                loss, acc = model.train_on_batch(X_train, y_train)
                if it % 10 == 0:
                    print(f"Batch {it}/{batch_count} - loss: {loss:.4f} - acc: {acc:.4f}")
                    
    print("\nSaving model to Model.h5...")
    model.save("Model.h5")
    
    print("\nEvaluating on validation set...")
    errors = 0
    val_batch_count = int(np.ceil(len(val_df) / BATCH_SIZE))
    for it in range(val_batch_count):
        X_val, y_val = get_batch(val_df, it * BATCH_SIZE, BATCH_SIZE, BASE_PATH, lencoder)
        if len(X_val) > 0:
            result = model.predict(X_val, verbose=0)
            preds = np.argmax(result, axis=1)
            errors += np.sum(preds != y_val)
            
    if len(val_df) > 0:
        acc = 100 * (len(val_df) - errors) / len(val_df)
        print(f"Errors: {errors}, Validation Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
