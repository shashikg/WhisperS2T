import os
import tempfile

# Get package path
BASE_PATH = os.path.dirname(__file__)

# Get the system temporary directory
SYS_TMP_PATH = tempfile.gettempdir()
WHISPER_S2T_SERVER_TMP_PATH = f"{SYS_TMP_PATH}/whisper_s2t_server"

# Create directories for data
RAW_AUDIO_PATH = f"{WHISPER_S2T_SERVER_TMP_PATH}/data/raw"
WAV_AUDIO_PATH = f"{WHISPER_S2T_SERVER_TMP_PATH}/data/wav"

# Clean data directory
# os.system(f"rm -rf {RAW_AUDIO_PATH}")
# os.system(f"rm -rf {WAV_AUDIO_PATH}")

os.makedirs(RAW_AUDIO_PATH, exist_ok=True)
os.makedirs(WAV_AUDIO_PATH, exist_ok=True)