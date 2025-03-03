import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import string
import pickle
from flask_cors import CORS
import os
import time

# Configure logging with rotating file handler
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, 'api.log')

handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)  # 10MB per file, keep 5 backups
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger('hate_speech_api')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Model definition
class LSTM_CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden_dim, cnn_hidden_dim, num_classes, dropout=0.5):
        super(LSTM_CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.conv1d = nn.Conv1d(lstm_hidden_dim*2, cnn_hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cnn_hidden_dim, num_classes)
    
    def forward(self, x):
        # Handle both float and long inputs
        if x.dtype == torch.float32:
            x = x.long()
        
        # Embedding layer
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # LSTM layer
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, lstm_hidden_dim*2]
        
        # Reshape for CNN: [batch_size, lstm_hidden_dim*2, seq_len]
        lstm_out = lstm_out.permute(0, 2, 1)
        
        # Apply CNN
        conv_out = F.relu(self.conv1d(lstm_out))  # [batch_size, cnn_hidden_dim, seq_len]
        
        # Pooling
        pooled = self.pool(conv_out).squeeze(-1)  # [batch_size, cnn_hidden_dim]
        
        # Dropout and final classification
        dropped = self.dropout(pooled)
        output = self.fc(dropped)  # [batch_size, num_classes]
        
        return output

# Load model and vocab outside the request context
model = None
vocab = None

# Preprocess text function (make sure this matches your training preprocessing)
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions (@user)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (#hashtag)
    text = re.sub(r'#\w+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    return text

# Function to convert text to sequence
def text_to_sequence(text, vocab_dict, max_len=100):
    tokens = text.split()
    sequence = [vocab_dict.get(token, 1) for token in tokens]  # Use 1 (<UNK>) for unknown words
    # Truncate or pad to fixed length
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    else:
        sequence += [0] * (max_len - len(sequence))
    return sequence

# Initialize the model and vocabulary
def init_app():
    global model, vocab
    try:
        logger.info("Loading model and vocabulary...")
        start_time = time.time()
        
        # Load vocabulary from file
        with open('./vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        logger.info(f"Loaded vocabulary with {len(vocab)} entries")
        
        # Load model parameters
        vocab_size = len(vocab)
        embed_dim = 100
        lstm_hidden_dim = 128
        cnn_hidden_dim = 128
        num_classes = 2
        dropout = 0.5
        
        # Initialize model
        model = LSTM_CNN(vocab_size, embed_dim, lstm_hidden_dim, cnn_hidden_dim, num_classes, dropout)
        model.load_state_dict(torch.load('./hate_speech_model_lstm_cnn.pth', map_location=device))
        model.to(device)
        model.eval()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Model and vocabulary loaded successfully in {elapsed_time:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Error loading model or vocabulary: {str(e)}", exc_info=True)
        return False

@app.route('/')
def index():
    logger.info("Home endpoint accessed")
    return "Hate Speech Detection API"

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    logger.info("Received prediction request")
    
    try:
        # Get request data
        data = request.json
        if not data or 'text' not in data:
            logger.warning("Invalid request format - missing 'text' field")
            return jsonify({'error': 'Invalid request format. Please provide text field.'}), 400
        
        text = data['text']
        logger.info(f"Processing text (first 100 chars): {text[:100]}")
        
        # Preprocess text
        text = preprocess_text(text)
        
        # Convert to sequence
        sequence = text_to_sequence(text, vocab, max_len=100)
        sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(sequence_tensor)
            pred_probs = F.softmax(outputs, dim=1)
            pred_label = torch.argmax(pred_probs, dim=1).item()
        
        # Get label and confidence score
        label = ['Non-Hate Speech', 'Hate Speech'][pred_label]
        score = float(pred_probs[0][pred_label])
        
        result = {
            'label': label,
            'score': score
        }
        
        elapsed_time = time.time() - start_time
        logger.info(f"Prediction completed: {label} with confidence {score:.4f} in {elapsed_time:.4f} seconds")
        
        return jsonify(result)
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error during prediction: {str(e)} after {elapsed_time:.4f} seconds", exc_info=True)
        return jsonify({'error': 'An error occurred during prediction'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    if model is None or vocab is None:
        logger.error("Health check failed: model or vocabulary not loaded")
        return jsonify({'status': 'error', 'message': 'Model or vocabulary not loaded'}), 500
    
    logger.info("Health check passed")
    return jsonify({'status': 'ok', 'message': 'API is healthy'})

@app.errorhandler(404)
def not_found(e):
    logger.warning(f"404 error: {request.path}")
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500 error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if init_app():
        logger.info("Starting Flask API server...")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    else:
        logger.critical("Failed to initialize the application")
