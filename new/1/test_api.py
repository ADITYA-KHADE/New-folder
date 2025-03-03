import requests
import json
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("api_tester")

def test_prediction(text):
    """Test the prediction endpoint with the given text"""
    url = "http://localhost:5000/predict"
    payload = {"text": text}
    
    try:
        logger.info(f"Sending prediction request with text: {text[:100]}...")
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Prediction: {result['label']} (confidence: {result['score']:.4f})")
            logger.info(f"Response time: {elapsed_time:.4f} seconds")
            return result
        else:
            logger.error(f"Error: Received status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        logger.error("Error: Could not connect to the server. Make sure it's running.")
        return None
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return None

def test_health_check():
    """Test the health check endpoint"""
    url = "http://localhost:5000/health"
    
    try:
        logger.info("Sending health check request...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Health check status: {result['status']}")
            return result
        else:
            logger.error(f"Error: Received status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error during health check: {str(e)}")
        return None

if __name__ == "__main__":
    logger.info("Starting API tests...")
    
    # Test health check
    health_result = test_health_check()
    
    # Test various text samples
    test_samples = [
        "I love the way people support each other online",
        "I hate that idiot and his stupid ideas",
        "The movie was terrible, I absolutely hated it",
        "These people are disgusting and should not be allowed in our country",
        "Everyone deserves respect and dignity regardless of their background"
    ]
    
    results = []
    for sample in test_samples:
        result = test_prediction(sample)
        if result:
            results.append((sample, result["label"], result["score"]))
    
    # Print summary
    logger.info("\n--- Test Summary ---")
    for idx, (sample, label, score) in enumerate(results, 1):
        logger.info(f"Sample {idx}: {sample[:50]}... -> {label} ({score:.4f})")
    
    logger.info("API testing completed")
