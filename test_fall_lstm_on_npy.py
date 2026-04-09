import numpy as np
import tensorflow as tf
import glob
import os
import h5py

SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 17
MODEL_PATH = "fall_lstm.h5"
DATA_DIR = "fall_sequences"

# ============================================================
# Rebuild model thủ công để tránh lỗi Lambda layer
# ============================================================
def build_model(input_shape=(30, 34)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

model = build_model()

# Load weights từ file .h5 cũ
try:
    model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
    print("✅ Load weights thành công")
except Exception as e:
    print(f"⚠️ Load weights lỗi: {e}")
    print("Thử load toàn bộ model với safe_mode=False...")
    import keras
    keras.config.enable_unsafe_deserialization()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)

# ============================================================
# Inference
# ============================================================
files = glob.glob(os.path.join(DATA_DIR, '*.npy'))
print(f"Tìm thấy {len(files)} file .npy")

for f in files:
    arr = np.load(f)  # (30, 17, 2)
    if arr.shape != (SEQUENCE_LENGTH, NUM_KEYPOINTS, 2):
        print(f"Bỏ qua {f}, shape {arr.shape} không hợp lệ")
        continue

    seq = arr.reshape((1, SEQUENCE_LENGTH, NUM_KEYPOINTS * 2))
    min_v = seq.min(axis=1, keepdims=True)
    max_v = seq.max(axis=1, keepdims=True)
    denom = max_v - min_v
    denom[denom == 0] = 1
    seq_norm = (seq - min_v) / denom

    pred = model.predict(seq_norm, verbose=0)
    fall_prob = float(pred[0][1])
    label = "🔴 FALL" if fall_prob > 0.5 else "🟢 NORMAL"
    print(f"{os.path.basename(f)} | fall_prob={fall_prob:.3f} | {label}")