
# Hate Speech Detection API

This API provides hate speech detection capabilities using a Bidirectional LSTM-CNN neural network model.

## Setup and Installation

1. Make sure all required dependencies are installed:

```bash
pip install flask torch numpy scikit-learn pandas flask-cors
```

2. The following files should be present in the directory:
   - `api.py` - The Flask API server
   - `hate_speech_model_lstm_cnn.pth` - The trained model
   - `vocab.pkl` - The vocabulary dictionary used by the model

## Running the API Server

1. Start the server:

```bash
python api.py
```

2. The server will run on `http://localhost:5000` by default

## API Endpoints

### Prediction Endpoint

**URL**: `/predict`

**Method**: POST

**Request Body**:
```json
{
  "text": "your text here"
}
```

**Response**:
```json
{
  "label": "Non-Hate Speech", // or "Hate Speech"
  "score": 0.9876 // confidence score
}
```

### Health Check Endpoint

**URL**: `/health`

**Method**: GET

**Response**:
```json
{
  "status": "ok",
  "message": "API is healthy"
}
```

## Logging

The API logs important events to `api.log` in the same directory. The logs include:
- Server startup information
- Model loading status
- Request processing details
- Errors and exceptions

The log files use a rotating file handler, keeping the last 5 log files with a maximum size of 10MB each.

## Testing the API

You can use the included `test_api.py` script to test the API:

```bash
python test_api.py
```

This will send several test requests and log the responses to both the console and `api_test.log`.

## Model Information

The model is a Bidirectional LSTM with CNN for hate speech detection, trained on labeled datasets. It achieves high accuracy in detecting hate speech in text content.

Key features:
- Preprocessing of text to remove URLs, mentions, hashtags, etc.
- Word embedding layer
- Bidirectional LSTM layers
- Convolutional layer for feature extraction
- Classification layer for binary prediction (Hate Speech / Non-Hate Speech)
