import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Layer
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

"""
Train LSTM phát hiện té ngã từ dữ liệu keypoints đã thu thập.
- Đọc dữ liệu .npy từ thư mục fall_sequences
- Nhãn: 'fall' = 1, 'normal' = 0
- Lưu model ra file fall_lstm.h5
"""

DATA_DIR = "fall_sequences"
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 17
SAVE_MODEL_PATH = "fall_lstm.h5"

# Đọc dữ liệu
X = []
y = []
for file in glob(os.path.join(DATA_DIR, '*.npy')):
    arr = np.load(file)  # (SEQUENCE_LENGTH, 17, 2)
    if arr.shape != (SEQUENCE_LENGTH, NUM_KEYPOINTS, 2):
        continue
    label = 1 if 'fall' in os.path.basename(file) else 0
    X.append(arr)
    y.append(label)
X = np.array(X)  # (samples, SEQUENCE_LENGTH, 17, 2)
y = np.array(y)
X = X.reshape((X.shape[0], SEQUENCE_LENGTH, NUM_KEYPOINTS * 2))  # (samples, SEQUENCE_LENGTH, 34)

# Normalize keypoints (min-max theo từng sequence)
def normalize_sequence(seq):
    # seq: (SEQUENCE_LENGTH, 34)
    min_v = seq.min(axis=0)
    max_v = seq.max(axis=0)
    denom = (max_v - min_v)
    denom[denom == 0] = 1  # tránh chia 0
    return (seq - min_v) / denom

X = np.array([normalize_sequence(s) for s in X])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# One-hot labels
y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

# Attention Layer


# Hàm reduce_sum_axis1 tách riêng để dùng custom_objects khi load model
def reduce_sum_axis1(xin):
    import tensorflow as tf
    return tf.reduce_sum(xin, axis=1)

# Attention block tách riêng để dùng custom_objects khi load model
def attention_block(inputs):
    attention = tf.keras.layers.Dense(1, activation='tanh')(inputs)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(inputs.shape[-1])(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)
    sent_representation = tf.keras.layers.Multiply()([inputs, attention])
    sent_representation = tf.keras.layers.Lambda(
        reduce_sum_axis1,
        output_shape=lambda s: (s[0], s[2])
    )(sent_representation)
    return sent_representation

# Model with Attention
inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, NUM_KEYPOINTS*2))
x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = attention_block(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train_cat, epochs=30, batch_size=16, validation_data=(X_test, y_test_cat))

# Save
model.save(SAVE_MODEL_PATH)
print(f"Đã lưu model: {SAVE_MODEL_PATH}")
