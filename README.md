# Voice to Text Sentiment Analysis (RNN/LSTM)

Mini project that takes speech audio, extracts MFCC features, and uses an
LSTM model to predict the emotion in the voice. The predicted emotion is
then mapped to a sentiment (positive / negative / neutral).

Dataset used: RAVDESS
Download: https://zenodo.org/record/1188976

## Setup

```
pip install -r requirements.txt
```

Download the RAVDESS dataset and place the extracted folder in this
directory, named `RAVDESS`. Update `DATASET_PATH` in the script if you
name it something else.

## Run

```
python speech_emotion_student.py
```

This will:
- load all `.wav` files from the dataset
- extract MFCC features
- train an LSTM model to classify emotion
- print test accuracy
- save a training accuracy/loss graph as `training_graphs.png`

## Predict on a new file

At the bottom of `speech_emotion_student.py`, uncomment:

```python
predict_emotion("test_audio.wav")
```

and set the path to any `.wav` file to get its predicted emotion and sentiment.
