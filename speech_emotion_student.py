"""
Voice to Text Sentiment Analysis using RNN (LSTM)
Mini Project - 2nd Year BTech

Idea: Take speech audio files, extract MFCC features from them,
and train an LSTM model to predict the emotion in the voice.
Then map that emotion to a sentiment (positive / negative / neutral).

Dataset used: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech)
Download from: https://zenodo.org/record/1188976
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -------------------------------------------------
# STEP 1: Setup
# -------------------------------------------------

DATASET_PATH = "RAVDESS"   # change this to your dataset folder path

# RAVDESS filenames look like: 03-01-06-01-02-01-12.wav
# the 3rd number is the emotion code
emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# simple mapping from emotion to sentiment
def emotion_to_sentiment(emotion):
    if emotion in ["happy", "calm", "surprised"]:
        return "positive"
    elif emotion == "neutral":
        return "neutral"
    else:
        return "negative"


# -------------------------------------------------
# STEP 2: Feature extraction function
# -------------------------------------------------

def extract_mfcc(file_path, max_len=100):
    # load audio file
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    # extract 40 MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc = mfcc.T   # shape becomes (time_steps, 40)

    # pad or cut so every audio file gives same size input
    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_len, :]

    return mfcc


# -------------------------------------------------
# STEP 3: Load dataset and create X, y
# -------------------------------------------------

X = []
y = []

print("Reading audio files and extracting features...")

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            # get emotion code from filename
            parts = file.split("-")
            emotion_code = parts[2]
            emotion = emotion_labels.get(emotion_code)

            if emotion is None:
                continue

            features = extract_mfcc(file_path)
            X.append(features)
            y.append(emotion)

X = np.array(X)
print("Total samples:", len(X))
print("Shape of one sample:", X[0].shape)

# -------------------------------------------------
# STEP 4: Encode labels
# -------------------------------------------------

le = LabelEncoder()
y_encoded = le.fit_transform(y)          # convert emotion names to numbers
y_categorical = to_categorical(y_encoded)  # one hot encoding

print("Emotion classes:", list(le.classes_))

# -------------------------------------------------
# STEP 5: Train-test split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# -------------------------------------------------
# STEP 6: Build the RNN (LSTM) model
# -------------------------------------------------

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# -------------------------------------------------
# STEP 7: Train the model
# -------------------------------------------------

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32
)

# -------------------------------------------------
# STEP 8: Evaluate the model
# -------------------------------------------------

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# plot accuracy and loss graphs
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig("training_graphs.png")
plt.show()

# -------------------------------------------------
# STEP 9: Predict emotion + sentiment for a new audio file
# -------------------------------------------------

def predict_emotion(file_path):
    features = extract_mfcc(file_path)
    features = np.expand_dims(features, axis=0)  # model expects batch dimension

    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    predicted_emotion = le.inverse_transform([predicted_index])[0]
    predicted_sentiment = emotion_to_sentiment(predicted_emotion)

    print("Predicted Emotion:", predicted_emotion)
    print("Predicted Sentiment:", predicted_sentiment)

    return predicted_emotion, predicted_sentiment


# example usage (uncomment and change path to test)
# predict_emotion("test_audio.wav")
