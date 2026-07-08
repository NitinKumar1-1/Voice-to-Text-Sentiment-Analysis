# Voice to Text Sentiment Analysis (RNN/LSTM)

Mini project that takes speech audio, extracts MFCC features, and uses an
LSTM model to predict the emotion in the voice. The predicted emotion is
mapped to a sentiment (positive / negative / neutral).

Dataset used: RAVDESS
Download: https://zenodo.org/record/1188976

## How to run (Google Colab)

1. Open `Speech_Emotion_LSTM.ipynb` in Google Colab.
2. Upload the RAVDESS dataset folder to your Google Drive.
3. Update `DATASET_PATH` in the notebook to point to that folder.
4. Run all cells.

This will:
- load all `.wav` files from the dataset
- extract MFCC features
- train an LSTM model to classify emotion
- print test accuracy
- plot training accuracy/loss graphs
- save the trained model as `speech_emotion_lstm_model.keras`

## Predict on a new file

In the last cell, uncomment:

```python
predict_emotion("test_audio.wav")
```

and set the path to any `.wav` file to get its predicted emotion and sentiment.

## Files

- `Speech_Emotion_LSTM.ipynb` - main notebook
- `requirements.txt` - dependencies (if running locally instead of Colab)
