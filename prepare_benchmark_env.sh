# Install packages
apt-get update
apt-get install -y libsndfile1 ffmpeg
pip install -U -r requirements.txt
pip install -U git+https://github.com/m-bain/whisperx.git
pip install -U -r benchmark_requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation

# Download dataset
rm -rf data.zip
wget https://github.com/shashikg/WhisperS2T/releases/download/v1.0.0/data.zip
unzip data.zip
rm -rf data.zip