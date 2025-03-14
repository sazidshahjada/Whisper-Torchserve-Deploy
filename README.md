# Torchserve Whisper

This document provides a comprehensive guide for the Whisper TorchServe project. It covers the project overview, directory structure, system requirements, setup, build and deployment instructions, and usage details.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [System Requirements](#system-requirements)
4. [Setup and Installation](#setup-and-installation)
5. [Building the Docker Image](#building-the-docker-image)
6. [Running the Server](#running-the-server)
7. [Model Archiving](#model-archiving)
8. [Audio Transcription Workflow](#audio-transcription-workflow)
9. [Testing and Bulk Request Examples](#testing-and-bulk-request-examples)
10. [Configuration Details](#configuration-details)
11. [Logging and Diagnostics](#logging-and-diagnostics)
12. [Future Improvements](#future-improvements)

---

## Overview

The Whisper TorchServe project is built for Dockerized deployment of [OpenAI’s Whisper](https://github.com/openai/whisper) model using TorchServe. The goal is to provide a scalable and GPU-enablement inference service for audio transcription. The project integrates a custom handler for preprocessing, inference, and postprocessing of audio input. It also ensures a smooth deployment pipeline by incorporating a model archiver, client scripts for testing, and robust configuration settings.

---

## Project Structure

The repository is organized with clear separation of concerns:

- **Root Files:**
  - `.gitignore` – Specifies files and directories to exclude from source control.
  - `Dockerfile` – Instructions for building the Docker image.
  - `README.md` – This documentation.
  - `requirements.txt` – Python dependencies for the project.
  - `serve.sh` – Shell script to start TorchServe.
  - Client and testing scripts:
    - `bulk_request.py`
    - `download_audio.py`
    - `download_model.py`
    - `request_train.py`
    - `test_client.py`
  - Sample audio files (e.g., `emma.mp3`, `harvard.wav`, etc.)

- **Configuration:**
  - `/config/config.properties` – TorchServe configuration.
  - `/config/model-config.yaml` – Additional model configuration parameters.

- **Model Archive:**
  - `/model-archiver/create_model_archive.sh` – Script to package the model and handler.
  - `/model-archiver/requirements.txt` – Model archive dependencies.

- **Custom Handler:**
  - `/handler/whisper_handler.py` – Implements the [WhisperHandler](handler/whisper_handler.py) class, which is responsible for:
    - **Preprocessing:** Converting input audio to mono, resampling, and optionally padding/trimming.
    - **Inference:** Running the Whisper model on processed audio (handles both full audio and chunked inputs).
    - **Postprocessing:** Formatting the transcription and clearing GPU cache.

- **Logs and Downloads:**
  - `/logs/` – Directory for various TorchServe logs.
  - `/downloads/` – For storing downloaded audio files.

- **Model Storage:**
  - `/whisper-models/` – Stores the pre-downloaded model weight file (e.g., `whisper_base.pth`).

---

## System Requirements

- **Hardware:**  
  A GPU-enabled system is recommended (Docker image uses a GPU-enabled base such as `pytorch/torchserve:latest-gpu`).

- **Software:**  
  - Docker  
  - Python 3 (with pip)
  - CUDA (when using GPU)
  - yt_dlp, torch, whisper, soundfile, librosa, numpy, etc.

---

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd whisper-torchserve
   ```

2. **Install Dependencies:**

   For local testing, install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Whisper Model:**

   Use the provided script to download the model (ensuring GPU usage if available):

   ```bash
   python download_model.py
   ```

   The model file will be saved under `/whisper-models/whisper_base.pth`.

---

## Building the Docker Image

To build the Docker image for the project, run:
   
```bash
sudo docker build -t whisper-torchserve .
```

This will build an image using the GPU-enabled TorchServe base image and install all required dependencies.

---

## Running the Server

After building the Docker image, run the server with the following command (ensuring GPU support is enabled):

```bash
sudo docker run --gpus all -p 8080:8080 -p 8081:8081 -p 8082:8082 whisper-torchserve
```

The `serve.sh` script ([serve.sh](serve.sh)) will:
- Create the logs directory.
- Verify the existence of the pre-downloaded model.
- Create the model archive using the script in [model-archiver/create_model_archive.sh](model-archiver/create_model_archive.sh).
- Start a TorchServe instance with the appropriate configurations from [config/config.properties](config/config.properties).

---

## Model Archiving

The model archive is created using the `torch-model-archiver` command in the [model-archiver/create_model_archive.sh](model-archiver/create_model_archive.sh) script. Key steps include:

- Installing model dependencies from [model-archiver/requirements.txt](model-archiver/requirements.txt).
- Verifying the existence of the custom handler: [handler/whisper_handler.py](handler/whisper_handler.py).
- Exporting the archive to `/home/model-server/model-store`.

Once archived, the file `whisper.mar` is used by TorchServe to serve the model.

---

## Audio Transcription Workflow

1. **Preprocessing:**
   - The [WhisperHandler.preprocess](handler/whisper_handler.py) method reads the audio bytes, converts multi-channel audio into mono, and resamples the audio to 16 kHz if needed.
   - Optionally, code for fixing audio duration (padding or trimming) is included in comments for future adjustments.

2. **Inference:**
   - The [WhisperHandler.inference](handler/whisper_handler.py) method supports both single audio inputs and chunked data.  
   - For each data segment, the audio is converted into a tensor and passed to the Whisper model, ensuring computation on the correct device (GPU/CPU).

3. **Postprocessing:**
   - The [WhisperHandler.postprocess](handler/whisper_handler.py) method formats the transcription result and clears GPU memory by invoking `torch.cuda.empty_cache()`.

---

## Testing and Bulk Request Examples

- **Single Request:**  
  Use [test_client.py](test_client.py) to send an individual audio file for transcription.
  
  ```bash
  python test_client.py <audio_file_path>
  ```

- **Bulk Requests:**  
  The [bulk_request.py](bulk_request.py) script allows concurrent transcription requests, which is useful for load testing.

  ```bash
  python bulk_request.py <audio_file1> <audio_file2> ...
  ```

---

## Configuration Details

- **TorchServe Configuration ([config/config.properties](config/config.properties)):**
  - **Inference Address:** `http://0.0.0.0:8080`
  - **Management Address:** `http://0.0.0.0:8081`
  - **Metrics Address:** `http://0.0.0.0:8082`
  - **Payload Settings:** Request/Response size increases (150MB) and extended timeouts (900 seconds) to handle large audio files.
  - **Performance:** Uses boosting settings like increased number of netty threads and job queue size.

- **Model Configuration ([config/model-config.yaml](config/model-config.yaml)):**
  - Specifies GPU usage, number of workers, batch size, and delay configurations.

---

## Logging and Diagnostics

- **Logging:**  
  The project includes comprehensive logging that records:
  - Environment details (Python version, CUDA information).
  - Model loading status and device placement.
  - Preprocessing steps and audio details (sample rate, duration).
  - Inference timing and errors.
  - GPU memory cleanup after inference.
  
  Logs are stored in `/logs/` (e.g., `access_log.log`, `model_log.log`, etc.) which help in debugging and performance analysis.

- **Monitoring:**  
  The TorchServe management endpoint and metrics address provide real-time monitoring capabilities.

---

## Future Improvements

- **Improved Chunking:**  
  Increase support for audio chunking, allowing more robust handling of long audio inputs.

- **Error Handling Enhancements:**  
  Refine error messages and fallback mechanisms for downloading and model loading.

- **Streamlined API:**  
  Further document and refine client API usage based on user feedback.

- **Performance Tuning:**  
  Experiment with batch size and worker configurations to better utilize hardware resources on different deployment scales.

---

## Conclusion

This documentation outlines the design, installation, and operational workflow of the Whisper TorchServe project. By following these guidelines, users and developers can effectively deploy and scale audio transcription services with GPU acceleration.

For more details, please refer to the individual file documentation and inline comments available in the repository.