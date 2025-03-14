import os
import torch
import whisper
import librosa
import soundfile as sf
import numpy as np
import logging
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

    def preprocess(self, data):
        """
        Preprocess audio data.
        Loads the complete audio, converts to mono and resamples if needed.
        """
        audio_data = data[0].get("audio")
        
        try:
            if isinstance(audio_data, (bytes, bytearray)):
                audio_bytes = BytesIO(audio_data)
                y, sr = sf.read(audio_bytes)
                if y.ndim > 1:  # Convert stereo to mono
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
                
                # Return the processed audio
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
            # If data is a list (which may result from chunking in other implementations)
            # process each chunk; otherwise, process the full audio.
            if isinstance(data, list):
                transcriptions = []
                for idx, chunk in enumerate(data):
                    logger.info(f"Processing chunk {idx+1}/{len(data)}")
                    # Process each chunk as-is.
                    audio_tensor = torch.tensor(chunk, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        result = whisper.transcribe(self.model, audio_tensor, language="en", temperature=0)
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