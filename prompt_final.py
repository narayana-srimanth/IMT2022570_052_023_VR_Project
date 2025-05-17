# Import required libraries for JSON, file operations, image encoding, API requests, progress tracking, and timing
import json
import os
import base64
import requests
from tqdm import tqdm  # For progress bar
import time
from google.generativeai import configure, GenerativeModel  # Google Gemini API
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable  # For API error handling

# Configuration variables
INPUT_FILE = 'cleaned_vqa_metadata_with_images.json'  # Input file with product metadata
OUTPUT_FILE = 'vqa_training_data.json'  # Output file for VQA training data
API_KEY = 'YOUR_API_KEY_HERE'  # Gemini API key (replace with actual key)
IMAGE_BASE_DIR = "abo-images-small/images/small"  # Directory containing product images
RETRY_ATTEMPTS = 3  # Number of retry attempts for failed API calls
DELAY = 3.0  # Delay (seconds) between API calls to avoid rate limiting

# Range variables for processing specific data subsets
START_INDEX = 3078  # Starting index (0-based, inclusive)
END_INDEX = 4500    # Ending index (exclusive)
APPEND_RESULTS = True  # Append results to existing output file if True

def encode_image(image_path):
    """Encode an image file to base64 string for API submission.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Base64-encoded string of the image, or None if encoding fails
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def generate_vqa_data(model, metadata, image_path, retry_attempts=RETRY_ATTEMPTS, delay=DELAY):
    """Generate Visual Question Answering (VQA) data for a product using the Gemini API.
    
    Args:
        model: Initialized Gemini GenerativeModel instance
        metadata: Product metadata dictionary
        image_path: Path to the product image
        retry_attempts: Number of retry attempts for API failures
        delay: Delay (seconds) between retries
    
    Returns:
        JSON object with image ID and VQA questions/answers, or None if generation fails
    """
    # Define prompt for the Gemini API to generate diverse VQA questions
    prompt = """You are an AI assistant helping to generate training data for a Visual Question Answering (VQA) model.
You are provided with:
- A product image
- Detailed product metadata (brand, style, color, features, description, etc.)

Your task is to generate diverse and meaningful questions that require both visual understanding and contextual reasoning from the metadata. The goal is to help train a robust VQA model that generalizes well to unseen product types and questions.

Use both the image and the metadata together to craft the questions. Make sure each question is visually answerable using the image while being enhanced by the metadata. Do not copy metadata text directly into answers — paraphrase or infer instead. Encourage variety in question types and phrasing. Avoid overfitting by ensuring questions are not repeated across images or overly templated.

Guidelines:
- Generate 2 to 3 diverse questions per image.
- Questions must be answerable based on the image, optionally supported by metadata.
- Keep answers short and specific (1 word max).
- Use a mix of question types as appropriate for the image:
  - Descriptive
  - Counting
  - Comparative
  - Color recognition
  - Function-based
  - Reasoning-based
-Ensure an increasing level of question complexity and reasoning


Output Format (strict JSON format):
{
  "image_id": "IMAGE_ID_HERE",
  "questions": [
    {
      "question": "QUESTION TEXT HERE",
      "answer": "ANSWER HERE"
    },
    {
      "question": "QUESTION TEXT HERE",
      "answer": "ANSWER HERE"
    }
  ]
}

Product Metadata:
"""
    prompt += json.dumps(metadata, indent=2)  # Append formatted metadata to prompt
    
    # Encode image to base64
    image_data = encode_image(image_path)
    if not image_data:
        return None
    
    # Prepare image data for API
    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": image_data
        }
    ]
    
    # Attempt API call with retries
    for attempt in range(retry_attempts):
        try:
            response = model.generate_content(
                contents=[
                    {"role": "user", "parts": [{"text": prompt}, {"inline_data": image_parts[0]}]}
                ],
                generation_config={
                    "temperature": 0.4,  # Control randomness
                    "max_output_tokens": 1024,  # Limit response length
                }
            )
            
            # Extract JSON from response (handle markdown code blocks)
            response_text = response.text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON response: {json_str}")
            else:
                print(f"No valid JSON found in response: {response_text}")
            
            # Retry if response is invalid
            if attempt < retry_attempts - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                
        except (ResourceExhausted, ServiceUnavailable) as e:
            print(f"API limit exceeded or service unavailable: {e}")
            if attempt < retry_attempts - 1:
                sleep_time = delay * (2 ** attempt)  # Exponential backoff
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Maximum retry attempts reached.")
                return None
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            if attempt < retry_attempts - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                return None
    
    return None

def append_to_json_file(data, filename):
    """Append VQA data to an existing JSON file or create a new one.
    
    Args:
        data: VQA data to append (list or single item)
        filename: Path to the output JSON file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read existing content if file exists
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    print(f"Error reading existing file {filename}. Creating new file.")
                    existing_data = []
        else:
            existing_data = []
        
        # Append new data
        if isinstance(data, list):
            existing_data.extend(data)
        else:
            existing_data.append(data)
            
        # Write back to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2)
            
        return True
    except Exception as e:
        print(f"Error appending to JSON file: {e}")
        return False

def main():
    """Main function to process product metadata and generate VQA training data."""
    # Configure Gemini API with provided key
    configure(api_key=API_KEY)
    model = GenerativeModel('gemini-1.5-flash')
    
    # Read input JSON file
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            try:
                # Attempt to load as single JSON object
                data = json.load(f)
                # Convert dictionary to list of single-item dictionaries
                if isinstance(data, dict):
                    data = [{k: v} for k, v in data.items()]
            except json.JSONDecodeError:
                # Try parsing as JSON Lines if single-object load fails
                f.seek(0)
                data = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line: {line[:50]}... - {e}")
            
            # Validate and adjust processing range
            start_idx = START_INDEX
            end_idx = END_INDEX if END_INDEX is not None else len(data)
            
            if start_idx < 0:
                start_idx = 0
            if end_idx > len(data):
                end_idx = len(data)
            if start_idx >= end_idx:
                print(f"Invalid range: start ({start_idx}) must be less than end ({end_idx})")
                return
                
            # Select data subset to process
            data_to_process = data[start_idx:end_idx]
                
            if not data_to_process:
                print("No valid data found in the specified range.")
                return
                
            print(f"Processing items from index {start_idx} to {end_idx-1} ({len(data_to_process)} items)")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    results = []
    
    # Process each metadata entry with progress bar
    for idx, item_data in enumerate(tqdm(data_to_process, desc="Processing items")):
        try:
            # Extract item ID and metadata
            item_id = list(item_data.keys())[0]
            metadata = item_data[item_id]
            
            # Get primary image path
            primary_image_key = item_id
            if primary_image_key in metadata.get('image_metadata', {}):
                image_info = metadata['image_metadata'][primary_image_key]
                image_path = os.path.join(IMAGE_BASE_DIR, image_info['path'])
                
                print(f"Processing item {start_idx + idx} (ID: {item_id})")
                
                # Generate VQA data using Gemini API
                vqa_data = generate_vqa_data(model, metadata, image_path, delay=DELAY)
                
                if vqa_data:
                    vqa_data['image_id'] = item_id  # Ensure correct image ID
                    results.append(vqa_data)
                    
                    # Delay to avoid rate limiting
                    time.sleep(DELAY)
            else:
                print(f"Image metadata not found for item {item_id}")
        except Exception as e:
            print(f"Error processing item: {e}")
    
    # Write results to output file
    try:
        if APPEND_RESULTS:
            success = append_to_json_file(results, OUTPUT_FILE)
            if success:
                print(f"Appended VQA data for {len(results)} items to {OUTPUT_FILE}")
        else:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Generated VQA data for {len(results)} items. Saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()
