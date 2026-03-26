# 🏛️ Landmark Detection using Deep Learning

A deep learning project that trains a **VGG19** Convolutional Neural Network to recognize and classify famous landmarks from images using the **Google Landmark V2** dataset.

---

## 📌 Project Overview

This project builds an image classification pipeline that can identify different landmarks (such as famous buildings, monuments, tourist attractions, etc.) from photographs. It uses a VGG19 architecture trained from scratch on a subset of the Google Landmark V2 dataset.

---

## 📁 Project Structure

```
Landmark-Detection/
├── landmark_detection.py   # Main script — data loading, model training, evaluation
├── requirements.txt        # Python dependencies
├── train.csv               # Dataset metadata (image IDs and landmark labels)
├── images/                 # Image directory (organized in subfolders)
├── Model.h5                # Saved trained model (generated after training)
├── class_distribution.png  # Class distribution histogram (generated after EDA)
└── README.md               # Project documentation
```

---

## 🧠 How the Code Works

The entire pipeline is inside **`landmark_detection.py`**. Here's a step-by-step explanation of what each part does:

### 1. Configuration (Top of File)

```python
CSV_PATH = "train.csv"
BASE_PATH = "./images/"
SAMPLES = 20000
BATCH_SIZE = 16
EPOCHS = 15
IMG_SIZE = (224, 224)
```

These are the global settings that control the behavior of the script:
- **CSV_PATH** — Path to the CSV file containing image IDs and their landmark labels.
- **BASE_PATH** — Root folder where all the images are stored.
- **SAMPLES** — Only the first 20,000 rows from the CSV are used (to keep training manageable).
- **BATCH_SIZE** — Number of images processed together in one training step.
- **EPOCHS** — Number of full passes through the training data.
- **IMG_SIZE** — All images are resized to 224×224 pixels to match VGG19's expected input.

---

### 2. `load_data(csv_path, samples)`

**Purpose:** Loads the training metadata from `train.csv`.

- Reads the CSV file using Pandas.
- Extracts two columns: `id` (the image filename) and `landmark_id` (the class/label).
- Limits the data to the first `SAMPLES` rows.
- Returns a cleaned DataFrame.

---

### 3. `perform_eda(df)`

**Purpose:** Performs Exploratory Data Analysis (EDA) — helps you understand the dataset before training.

- Prints the total size of the dataset and the number of unique landmark classes.
- Counts how many images belong to each class and prints the top 10 most frequent classes.
- Reports how many classes have very few images (≤5 or 6–10), which can be a problem for training.
- Saves a histogram of the class distribution to **`class_distribution.png`**.

---

### 4. `get_image_path(fname, base_path)` and `get_image_and_label(num, df, base_path)`

**Purpose:** Retrieves an image and its label from the dataset.

- The Google Landmark dataset organizes images into nested folders using the first 3 characters of the filename. For example, image `abc123.jpg` would be stored at `images/a/b/c/abc123.jpg`.
- `get_image_path()` builds this folder path from the image ID.
- `get_image_and_label()` reads the actual image file using OpenCV. If the image is missing or corrupted, it returns a black placeholder image so the training doesn't crash.

---

### 5. `get_batch(dataframe, start, batch_size, base_path, lencoder)`

**Purpose:** Creates a batch of images and labels for training.

- Loads a group of images starting from index `start`.
- Resizes each image to 224×224 pixels.
- Normalizes pixel values from `[0, 255]` to `[0.0, 1.0]` (neural networks train better with small numbers).
- Encodes the text labels into integer IDs using `LabelEncoder`.
- Returns NumPy arrays of images and labels ready for the model.

---

### 6. `build_model(num_classes)`

**Purpose:** Constructs the deep learning model.

- Uses **VGG19** as the base architecture — a well-known CNN with 19 layers that is excellent at image feature extraction.
- Initializes VGG19 **without pre-trained weights** (`weights=None`), meaning all parameters are learned from scratch.
- Adds a **Flatten** layer to convert 2D feature maps into a 1D vector.
- Adds **BatchNormalization** to stabilize and accelerate training.
- Adds **Dropout (50%)** to prevent overfitting by randomly disabling neurons during training.
- Adds a final **Dense layer** with `softmax` activation — this outputs a probability for each landmark class.
- Compiles the model with:
  - **Optimizer:** RMSprop (learning rate = 0.0001, momentum = 0.09)
  - **Loss:** Sparse Categorical Crossentropy (standard for multi-class classification)
  - **Metric:** Accuracy

---

### 7. `main()` — The Training Pipeline

**Purpose:** Orchestrates the entire workflow.

#### Step A: Load & Analyze Data
- Calls `load_data()` to read the CSV.
- Calls `perform_eda()` to generate statistics and plots.

#### Step B: Encode Labels
- Fits a `LabelEncoder` on all unique landmark IDs so the model can work with integer labels.

#### Step C: Build Model
- Calls `build_model()` and prints a summary of the architecture.

#### Step D: Split Data
- Splits the dataset into **80% training** and **20% validation** using random sampling.

#### Step E: Train the Model
- Loops through `EPOCHS` (15 full passes through the data).
- In each epoch, shuffles the training data and processes it in batches of 16.
- Uses `model.train_on_batch()` to update model weights one batch at a time.
- Prints loss and accuracy every 10 batches so you can monitor progress.

#### Step F: Save Model
- Saves the fully trained model to **`Model.h5`** so it can be reused later without retraining.

#### Step G: Evaluate on Validation Set
- Runs predictions on the 20% validation data.
- Compares predicted classes against ground truth labels.
- Prints the total number of errors and the overall **validation accuracy**.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11 or 3.12 (TensorFlow does not support Python 3.14 yet)
- The Google Landmark V2 dataset

### Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
| Library | Purpose |
|---|---|
| `numpy` | Numerical array operations |
| `pandas` | Data loading and manipulation |
| `tensorflow` | Deep learning framework (includes Keras) |
| `keras` | High-level neural network API |
| `opencv-python` | Image reading and resizing |
| `matplotlib` | Plotting charts and histograms |
| `Pillow` | Image file handling |
| `scikit-learn` | Label encoding utilities |

### Run Training

```bash
python landmark_detection.py
```

This will:
1. Load and analyze the dataset
2. Build the VGG19 model
3. Train for 15 epochs
4. Save the model to `Model.h5`
5. Print validation accuracy

---

## 📊 Dataset

- **Source:** [Google Landmark Recognition V2](https://www.kaggle.com/c/landmark-recognition-2019)
- **Contents:** Images of landmarks with corresponding label IDs
- **Format:** A `train.csv` file mapping image IDs to landmark IDs, and an `images/` folder containing the actual image files organized in nested subdirectories.

---

## 📈 Model Architecture

```
VGG19 (19-layer CNN, no pre-trained weights)
    ↓
Flatten
    ↓
BatchNormalization
    ↓
Dropout (50%)
    ↓
Dense (softmax) → Predicts landmark class
```

---

## 📝 License

This project is open source and available for educational purposes.
