import os
import time
import argparse
from whisper_s2t.speech_segmenter import SpeechSegmenter

from . import pullTask, finishTask
from ...logger import Logger

VAD_WAIT_TIME = float(os.getenv("VAD_WAIT_TIME", "0.1"))

def start_worker(device):
    Logger.info(f"Started VAD Model!")
    speech_segmenter_model = SpeechSegmenter(device=device)
    while True:
        job_id, input_file = pullTask()

        if job_id is None:
            time.sleep(VAD_WAIT_TIME)
        else:
            Logger.info(f"[VAD] Processing job: {job_id}!")
            start_ends, _ = speech_segmenter_model(input_file=input_file)
            finishTask(job_id, start_ends)
            Logger.info(f"[VAD] Processed job: {job_id}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='whisper_s2t_server_vad_worker')
    parser.add_argument('--device', default='cpu', help='cpu/cuda')

    args = parser.parse_args()
    start_worker(args.device)