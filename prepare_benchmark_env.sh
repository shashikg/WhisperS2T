# Install packages
apt-get update
apt-get install -y libsndfile1 ffmpeg
pip install -U -r requirements.txt
pip install -U git+https://github.com/m-bain/whisperx.git
pip install -U -r benchmark_requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation

# Download dataset
gdown 1cBq27gvy_a1HywF_mAug2zOTZGearhhu
unzip data.zip