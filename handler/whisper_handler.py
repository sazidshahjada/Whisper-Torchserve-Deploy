import os
import torch
import whisper
import librosa
import soundfile as sf
import numpy as np
import logging
import subprocess
import tempfile
from io import BytesIO
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class WhisperHandler(BaseHandler):
    def __init__(self):
        super(WhisperHandler, self).__init__()
        self.model = None
        self.initialized = False
        self.model_size = "base"

    def initialize(self, context):
        """
        Initialize model with explicit GPU configuration.
        """
        self.manifest = context.manifest
        properties = context.system_properties
        
        # Log environment details
        logger.info(f"Python version: {properties.get('python_version')}")
        logger.info(f"Torch version: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda")
        else:
            logger.warning("CUDA not available. Using CPU.")
            self.device = torch.device("cpu")
            
        logger.info(f"Using device: {self.device}")
        
        try:
            # Define paths to check for the pre-downloaded model
            model_paths = [
                f"/home/model-server/whisper-models/whisper_{self.model_size}.pth",
                os.path.join(os.getcwd(), f"whisper-models/whisper_{self.model_size}.pth"),
                os.path.join(properties.get("model_dir", ""), f"whisper_{self.model_size}.pth"),
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                    logger.info(f"Found saved model file at {model_path} ({os.path.getsize(model_path)/1024/1024:.2f} MB)")
                    try:
                        self.model = whisper.load_model(self.model_size, in_memory=True, device=self.device)
                        state_dict = torch.load(model_path, map_location=self.device)
                        self.model.load_state_dict(state_dict)
                        self.model = self.model.to(self.device)
                        logger.info(f"Successfully loaded model from {model_path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load model from {model_path}: {e}")
            
            if not model_loaded:
                logger.warning("No valid saved model found. Downloading model (this may take a while)...")
                self.model = whisper.load_model(self.model_size, device=self.device)
                save_path = "/home/model-server/whisper-models/whisper_base.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                logger.info(f"Saving model to {save_path}")
                torch.save(self.model.state_dict(), save_path)
                self.model = self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            for name, param in list(self.model.named_parameters())[:1]:
                logger.info(f"Model parameter {name} is on device: {param.device}")
            
            self.initialized = True
            logger.info("Model initialization complete")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def extract_audio_from_file(self, file_bytes):
        """
        Extract audio from any file (audio or video) using ffmpeg.
        
        Explanation:
        - Writes the file bytes to a temporary file.
        - Uses ffmpeg to convert the input file into a WAV file with 16000 Hz sample rate and mono channel.
        - Reads the WAV file with soundfile and then cleans up temporary files.
        This method should work regardless of the input file format.
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_input:
                tmp_input.write(file_bytes)
                input_path = tmp_input.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                output_path = tmp_audio.name

            command = [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000", "-ac", "1",
                "-f", "wav", output_path
            ]
            logger.info(f"Extracting audio using ffmpeg: {' '.join(command)}")
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr.decode()}")
                raise Exception("Failed to extract audio from input file")
            
            y, sr = sf.read(output_path)
            os.remove(input_path)
            os.remove(output_path)
            return y, sr
        except Exception as e:
            logger.error(f"Error during audio extraction: {e}")
            raise e

    def preprocess(self, data):
        """
        Preprocess audio data.
        Loads the complete audio (or other file formats), converts to mono and resamples if needed.
        If the input file is not recognized as native audio (WAV, Ogg, MP3, FLAC), 
        the file is processed using ffmpeg to extract the audio.
        """
        audio_data = data[0].get("audio")
        
        try:
            if isinstance(audio_data, (bytes, bytearray)):
                audio_bytes = BytesIO(audio_data)
                head = audio_bytes.read(12)
                audio_bytes.seek(0)
                
                # Check for known native audio headers:
                # WAV files usually start with "RIFF",
                # Ogg files with "OggS",
                # MP3 files might start with "ID3",
                # FLAC files start with "fLaC".
                if head.startswith(b"RIFF") or head.startswith(b"OggS") or head[:3] == b"ID3" or head.startswith(b"fLaC"):
                    logger.info("Detected native audio file format.")
                    y, sr = sf.read(audio_bytes)
                else:
                    logger.info("Input file format not recognized as native audio; attempting extraction via ffmpeg.")
                    y, sr = self.extract_audio_from_file(audio_bytes.read())
                    
                if y.ndim > 1:  # Convert stereo to mono if needed.
                    y = y.mean(axis=1)
                logger.info(f"Original sample rate: {sr}")
                if sr != 16000:
                    logger.info(f"Resampling from {sr} to 16000")
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    sr = 16000
                    
                duration = len(y) / sr
                logger.info(f"Audio duration: {duration} seconds")
                
                # NEW CODE: Uncomment below to fix the audio length to FIXED_DURATION_SECONDS.
                # FIXED_DURATION_SECONDS = 30  # Set desired fixed duration in seconds.
                # required_samples = int(FIXED_DURATION_SECONDS * sr)
                # if len(y) < required_samples:
                #     pad_length = required_samples - len(y)
                #     logger.info(f"Audio is shorter than {FIXED_DURATION_SECONDS} seconds; padding with {pad_length} zeros")
                #     y = np.pad(y, (0, pad_length), mode='constant')
                # elif len(y) > required_samples:
                #     logger.info(f"Audio is longer than {FIXED_DURATION_SECONDS} seconds; trimming to required length")
                #     y = y[:required_samples]
                # logger.info(f"Fixed audio length: {len(y)/sr:.2f} seconds")
                
                return y
            else:
                return audio_data
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise e

    def inference(self, data):
        """
        Run inference on the full (untrimmed) audio.
        """
        logger.info(f"Running inference on {self.device}")

        try:
            # If the data is not already a list, split the full audio into chunks.


            # if not isinstance(data, list):
            #     logger.info("Full audio provided, splitting into chunks...")
            #     # Define fixed chunk duration in seconds.
            #     chunk_duration_seconds = 30  # adjust as needed
            #     sample_rate = 16000  # Expected sample rate after preprocessing.
            #     chunk_length = int(chunk_duration_seconds * sample_rate)
            #     total_length = len(data)
            #     num_chunks = (total_length + chunk_length - 1) // chunk_length
            #     logger.info(f"Audio length: {total_length} samples. Splitting into {num_chunks} chunk(s) of {chunk_length} samples each.")
                
            #     chunks = []
            #     for i in range(num_chunks):
            #         start = i * chunk_length
            #         end = min((i + 1) * chunk_length, total_length)
            #         chunk = data[start:end]
            #         chunks.append(chunk)
            #     data = chunks


            # process each chunk; otherwise, process the full audio.
            if isinstance(data, list):
                transcriptions = []
                for idx, chunk in enumerate(data):
                    logger.info(f"Processing chunk {idx+1}/{len(data)}")
                    # Process each chunk as-is.
                    audio_tensor = torch.tensor(chunk, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        result = self.model.transcribe(audio_tensor, language="en", temperature=0)
                    transcriptions.append(result['text'])
                final_text = " ".join(transcriptions)
                logger.info("Inference complete for all chunks")
                return final_text
            else:
                logger.info("Processing full audio for whisper model without trimming")
                audio_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    result = whisper.transcribe(self.model, audio_tensor, language="en", temperature=0)
                logger.info("Inference complete")
                return result['text']
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise e

    def postprocess(self, inference_output):
        """
        Return the transcription results and clean up GPU memory.
        """
        result_str = str(inference_output)
        logger.info(f"Transcription result: {result_str[:30]}...")
        # Release unused cached GPU memory.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Emptied GPU cache after inference")
        return [{"transcription": inference_output}]