import requests
import json
import sys
import os
import time

def transcribe_audio(file_path, server_url="http://localhost:8080/predictions/whisper"):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return None
    
    # Read the audio file as binary data
    with open(file_path, 'rb') as file:
        files = {'audio': (os.path.basename(file_path), file, 'application/octet-stream')}
        
        print(f"Sending request to {server_url} with file {file_path}")
        file_size_mb = os.path.getsize(file_path) / (1024*1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Send with longer timeout (900 seconds)
        start_time = time.time()
        try:
            response = requests.post(server_url, files=files, timeout=900)
            process_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"Transcription completed in {process_time:.2f} seconds")
                return result
            else:
                print(f"Error: Server returned status code {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"Error: Request timed out after {time.time() - start_time:.2f} seconds")
            return None
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <audio_file_path>")
        sys.exit(1)
    
    result = transcribe_audio(sys.argv[1])
    if result:
        print("Transcription result:")
        print(json.dumps(result, indent=2))