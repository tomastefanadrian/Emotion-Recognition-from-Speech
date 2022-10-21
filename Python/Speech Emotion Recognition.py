from random import sample
import time

import os
import cv2
import pandas as pd
import numpy as np

import imageio
import tensorflow as tf
import librosa
import librosa.display
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal.windows import hamming

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score

import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.layers import InputLayer, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

DATASET_DIR = "Datasets/EmoDB"
SPECTROGRAM_DIR = "Spectrograms/EmoDB/Log"
TF_RECORDS_DIR = "TFRecords/EmoDB"
TF_RECORDS_NAME = "EmoDB_log.tfrecords"
MODEL_DIR = "Models"
MODEL_NAME = "EmoDB_log.h5"
SAMPLE_RATE = 16000
BATCH_SIZE = 32
EPOCHS = 13
RANDOM_SEED = 42  # for reproducibility


def create_dataframe_emodb():
    EMOTION_DICT_EMODB = {'W': 'anger', 'L': 'boredom', 'E': 'disgust', 'A': 'fear', 'F': 'happiness', 'T': 'sadness', 'N': 'neutral'}
    if DATASET_DIR != "Datasets/EmoDB":
        raise Exception("DATASET_DIR must be set to 'Datasets/EmoDB' for EmoDB dataset")
    file_person, file_gender, file_emotion, file_path = [], [], [], []
    file_list = os.listdir(DATASET_DIR)
    for file in file_list:
        person = int(file[0:2])
        gender = 'male' if person in [3, 10, 11, 12, 15] else 'female'
        emotion = EMOTION_DICT_EMODB[file[5]]
        file_person.append(person)
        file_gender.append(gender)
        file_emotion.append(emotion)
        file_path.append(os.path.join(DATASET_DIR, file))
    file_dict = {'person': file_person, 'gender': file_gender, 'emotion': file_emotion, 'path': file_path}
    emodb_df = pd.DataFrame.from_dict(file_dict)
    return emodb_df


def create_dataframe_emoiit():
    if DATASET_DIR != "Datasets/EMO-IIT":
        raise Exception("DATASET_DIR must be set to 'Datasets/EMO-IIT' for EMO-IIT dataset")
    file_emotion, file_path = [], []
    emotion_dir_list = os.listdir(DATASET_DIR)
    for emotion_dir in emotion_dir_list:
        file_list = os.listdir(os.path.join(DATASET_DIR, emotion_dir))
        for file in file_list:
            if file.endswith('.wav'):
                file_emotion.append(emotion_dir)
                file_path.append(os.path.join(DATASET_DIR, emotion_dir, file))
    file_dict = {'emotion': file_emotion, 'path': file_path}
    emoiit_df = pd.DataFrame.from_dict(file_dict)
    emoiit_df = pd.DataFrame(shuffle(emoiit_df, random_state=RANDOM_SEED), columns=emoiit_df.columns).reset_index(drop=True, inplace=False)
    return emoiit_df


def preprocess_dataset(ser_df, dataset_type, ohe=None):
    audio_block_list = []
    emotion_list = []
    for row in tqdm(ser_df.itertuples(), desc=f"Preprocessing audio files dataset - {dataset_type}", total=len(ser_df)):
        data, _ = librosa.load(row.path, sr=SAMPLE_RATE)
        if data.shape[0] < SAMPLE_RATE:
            data = np.pad(data, (0, SAMPLE_RATE - data.shape[0]), 'constant')
        frames = librosa.util.frame(data, frame_length=SAMPLE_RATE, hop_length=int(SAMPLE_RATE/100)).T
        for frame in frames:
            audio_block_list.append(frame)
            emotion_list.append(row.emotion)
    audio_block_list = np.array(audio_block_list)
    emotion_list = np.array(emotion_list)
    if ohe is None:
        ohe = OneHotEncoder(categories='auto', sparse=False)
        emotion_list = ohe.fit_transform(emotion_list[:, np.newaxis])
    else:
        emotion_list = ohe.transform(emotion_list[:, np.newaxis])
    return audio_block_list, emotion_list, ohe


def create_spectrogram_log(data, sr):
    X = np.abs(librosa.stft(data, window=hamming(int(np.round(sr / 1000) * 32)), n_fft=int(np.round(sr / 1000) * 32), hop_length=int(np.round(sr / 1000) * 4)))
    Xdb = librosa.amplitude_to_db(np.abs(X), ref=np.max)
    # plt.title('Spectrogram - Logarithmic', size=15)
    # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', cmap='jet', n_fft=int(np.round(sr/1000) * 32), hop_length=int(np.round(sr/1000) * 4))
    # plt.colorbar()
    # plt.show()
    cmap = plt.cm.jet
    # norm = plt.Normalize(vmin=np.amin(Xdb), vmax=np.amax(Xdb))
    norm = plt.Normalize(vmin=-90, vmax=-7)
    image = cmap(norm(Xdb))
    image = image[:, :, :3]
    image = cv2.normalize(src=np.flip(image, axis=0), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    return image


def create_spectrogram_linear(data, sr):
    X = np.abs(librosa.stft(data, window=hamming(int(np.round(sr / 1000) * 32)), n_fft=int(np.round(sr / 1000) * 32), hop_length=int(np.round(sr / 1000) * 4)))
    # plt.title('Spectrogram - Linear', size=15)
    # librosa.display.specshow(X, sr=sr, x_axis='time', y_axis='hz', cmap='jet', n_fft=int(np.round(sr/1000) * 32), hop_length=int(np.round(sr/1000) * 4))
    # plt.colorbar()
    # plt.show()
    cmap = plt.cm.jet
    # norm = plt.Normalize(vmin=-150, vmax=-27)
    norm = plt.Normalize(vmin=np.amin(X), vmax=np.amax(X))
    image = cmap(norm(X))
    image = image[:, :, :3]
    image = cv2.normalize(src=np.flip(image, axis=0), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    return image


def create_spectrogram_mel(data, sr):
    Xmel = np.abs(librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128, window=hamming(int(np.round(sr / 1000) * 32)), n_fft=int(np.round(sr / 1000) * 32), hop_length=int(np.round(sr / 1000) * 4)))
    X_mel_db = librosa.amplitude_to_db(Xmel, ref=np.max)
    # plt.title('Spectrogram - Mel', size=15)
    # librosa.display.specshow(X_mel_db, sr=sr, x_axis='time', y_axis='mel', cmap='jet')
    # plt.colorbar()
    # plt.show()
    cmap = plt.cm.jet
    # norm = plt.Normalize(vmin=np.amin(X_mel_db), vmax=np.amax(X_mel_db))
    norm = plt.Normalize(vmin=-90, vmax=-7)
    image = cmap(norm(X_mel_db))
    image = image[:, :, :3]
    image = cv2.normalize(src=np.flip(image, axis=0), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    return image


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example_train(image, path, emotion_id):
    feature = {
        "image": image_feature(image),
        "path": bytes_feature(path),
        "emotion_id": float_feature(emotion_id),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_example_test(image, path, emotion_id, sample_weight):
    feature = {
        "image": image_feature(image),
        "path": bytes_feature(path),
        "emotion_id": float_feature(emotion_id),
        "sample_weight": float_feature(sample_weight),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_train(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "emotion_id": tf.io.FixedLenFeature([7], tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_png(example["image"], channels=3)
    return example


def parse_tfrecord_test(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "emotion_id": tf.io.FixedLenFeature([7], tf.float32),
        "sample_weight": tf.io.FixedLenFeature([1], tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_png(example["image"], channels=3)
    return example


def create_spectrogram_dataset(audio_block_list, emotion_list, sr, create_spectrogram, sample_weight=None, dataset_type="train"):
    if dataset_type not in ["train", "dev", "test"]:
        raise ValueError("dataset_type must be 'train', 'dev' or 'test'")
    with tf.io.TFRecordWriter(os.path.join(TF_RECORDS_DIR, dataset_type, TF_RECORDS_NAME)) as writer:
        for index, block in enumerate(tqdm(audio_block_list, desc=f"Creating Spectrogram Dataset - {dataset_type}", total=audio_block_list.shape[0])):
            image = create_spectrogram(block, sr)
            image_path = os.path.join(f"{SPECTROGRAM_DIR}", dataset_type, f"{index:05d}.png")
            imageio.imsave(image_path, image)
            image = tf.io.decode_png(tf.io.read_file(image_path))
            if dataset_type == "train":
                example = create_example_train(image, image_path, emotion_list[index])
            else:
                if sample_weight is None:
                    raise ValueError("sample_weight must be provided for test dataset")
                else:
                    example = create_example_test(image, image_path, emotion_list[index], np.expand_dims(sample_weight[index], axis=0))
            writer.write(example.SerializeToString())


def prepare_sample_train(features):
    image = preprocess_input(tf.cast(features["image"], tf.float32))
    return image, features["emotion_id"]


def prepare_sample_test(features):
    image = preprocess_input(tf.cast(features["image"], tf.float32))
    sample_weight = tf.squeeze(features["sample_weight"])
    return image, features["emotion_id"], sample_weight


def get_dataset(filename, batch_size, dataset_type="train"):
    if dataset_type not in ["train", "dev", "test"]:
        raise ValueError("dataset_type must be 'train', 'dev' or 'test'")
    AUTOTUNE = tf.data.AUTOTUNE
    if dataset_type == "train":
        dataset = (
            tf.data.TFRecordDataset(filename, num_parallel_reads=AUTOTUNE)
            .map(parse_tfrecord_train, num_parallel_calls=AUTOTUNE)
            .map(prepare_sample_train, num_parallel_calls=AUTOTUNE)
            .shuffle(batch_size * 10, seed=RANDOM_SEED)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )
    else:
        dataset = (
            tf.data.TFRecordDataset(filename, num_parallel_reads=AUTOTUNE)
            .map(parse_tfrecord_test, num_parallel_calls=AUTOTUNE)
            .map(prepare_sample_test, num_parallel_calls=AUTOTUNE)
            .shuffle(batch_size * 10, seed=RANDOM_SEED)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )
    return dataset


def create_ser_model():
    vgg16 = VGG16(weights="imagenet")
    model = Model(inputs=vgg16.input, outputs=Dense(7, activation="softmax", name="emotion")(vgg16.get_layer("fc2").output))
    optimizer = tf.optimizers.SGD(learning_rate=0.0001, decay=0.0001, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], weighted_metrics=['accuracy'])
    return model


def plot_history(history, model_name):
    fig, axs = plt.subplots(2, 1, figsize=(15, 15))
    fig.suptitle(model_name, size=20)
    axs[0].plot(history.history['loss'])
    axs[0].title.set_text('Training Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[1].plot(history.history['accuracy'])
    axs[1].title.set_text('Training Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    plt.show()


def get_run_logdir(root_logdir):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


ser_df = create_dataframe_emodb()
ser_df_train, ser_df_test = train_test_split(ser_df, test_size=0.2, stratify=ser_df.emotion, random_state=RANDOM_SEED)
ser_df_dev, ser_df_test = train_test_split(ser_df_test, test_size=0.5, stratify=ser_df_test.emotion, random_state=RANDOM_SEED)
audio_block_list_train, emotion_list_train, ohe = preprocess_dataset(ser_df_train, "train")
audio_block_list_dev, emotion_list_dev, _ = preprocess_dataset(ser_df_dev, "dev", ohe)
audio_block_list_test, emotion_list_test, _ = preprocess_dataset(ser_df_test, "test", ohe)

cls_weight = class_weight.compute_class_weight(class_weight='balanced', classes=ohe.categories_[0], y=ohe.inverse_transform(emotion_list_train).flatten())
cls_weight_dict = dict(zip(ohe.categories_[0], cls_weight))
val_sample_weight = class_weight.compute_sample_weight(class_weight=cls_weight_dict, y=ohe.inverse_transform(emotion_list_dev).flatten())
test_sample_weight = class_weight.compute_sample_weight(class_weight=cls_weight_dict, y=ohe.inverse_transform(emotion_list_test).flatten())

create_spectrogram_dataset(audio_block_list_train, emotion_list_train, sr=SAMPLE_RATE, create_spectrogram=create_spectrogram_log, dataset_type="train")
create_spectrogram_dataset(audio_block_list_dev, emotion_list_dev, sr=SAMPLE_RATE, create_spectrogram=create_spectrogram_log, sample_weight=val_sample_weight, dataset_type="dev")
create_spectrogram_dataset(audio_block_list_test, emotion_list_test, sr=SAMPLE_RATE, create_spectrogram=create_spectrogram_log, sample_weight=test_sample_weight, dataset_type="test")

train_dataset = get_dataset(os.path.join(TF_RECORDS_DIR, "train", TF_RECORDS_NAME), batch_size=BATCH_SIZE, dataset_type="train")
dev_dataset = get_dataset(os.path.join(TF_RECORDS_DIR, "dev", TF_RECORDS_NAME), batch_size=BATCH_SIZE, dataset_type="dev")
test_dataset = get_dataset(os.path.join(TF_RECORDS_DIR, "test", TF_RECORDS_NAME), batch_size=BATCH_SIZE, dataset_type="test")

ser_model = create_ser_model()
ser_model.summary()
run_logdir = get_run_logdir(root_logdir=os.path.join(os.curdir, "logs\\fit\\"))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, MODEL_NAME), verbose=1, monitor='accuracy', save_best_only=True, mode='auto')
history = ser_model.fit(x=train_dataset, class_weight=dict(enumerate(cls_weight)), epochs=EPOCHS, verbose=1, callbacks=[tensorboard_cb, checkpoint], validation_data=dev_dataset)
plot_history(history, model_name="Model EmoDB: Log Spectrogram")

print("\nTest set results")
model = load_model(os.path.join(MODEL_DIR, MODEL_NAME))
results = model.evaluate(test_dataset)
print(f"Test loss: {results[0]}\nTest accuracy: {results[1]}\nTest weighted accuracy: {results[2]}")

test_dataset_slices = np.array(list(test_dataset.as_numpy_iterator()), dtype=object).T
y_true = test_dataset_slices[1][0]
for i in range(1, len(test_dataset_slices[1])):
    y_true = np.concatenate((y_true, test_dataset_slices[1][i]))
y_true = np.argmax(y_true, axis=1)
pred = model.predict(test_dataset)
y_pred = np.argmax(pred, axis=1)

print(classification_report(y_true, y_pred, target_names=ohe.categories_[0]))
confusion_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(11, 11))
sns.heatmap(confusion_mat, annot=True, fmt="d", xticklabels=ohe.categories_[0], yticklabels=ohe.categories_[0], cmap="Blues", linewidths=0.5)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
