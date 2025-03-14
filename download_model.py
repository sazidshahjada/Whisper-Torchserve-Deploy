import whisper
import os
import torch

MODEL_SIZE = "base"  # Options: "tiny", "base", "small", "medium", "large"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper-models")

def main():
    print(f"Downloading whisper {MODEL_SIZE} model...")
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Created directory: {MODEL_DIR}")
        used_model_dir = MODEL_DIR
    except PermissionError:
        print(f"Permission denied when creating {MODEL_DIR}")
        local_model_dir = os.path.join(os.getcwd(), "whisper-models")
        os.makedirs(local_model_dir, exist_ok=True)
        used_model_dir = local_model_dir

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"CUDA available: {cuda_available}, using {device}")

    print(f"Starting model download to {used_model_dir}...")
    model = whisper.load_model(MODEL_SIZE, device=device)
    model_path = os.path.join(used_model_dir, f"whisper_{MODEL_SIZE}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()