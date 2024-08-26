import os
import time
import traceback
from pebble import concurrent

from . import RAW_AUDIO_PATH, WAV_AUDIO_PATH

from .logger import Logger
from .models import asr, vad
from .utils import pullTask, updateJobStatus, finishTask, failedTask
from .audio import convert_to_wav16, load_audio, get_segmented_audio_signal

MAX_CONCURRENT_REQ = int(os.getenv("MAX_CONCURRENT_REQ", "8"))
WAIT_TIME_TO_FINISH = float(os.getenv("WAIT_TIME_TO_FINISH", "1.0"))
WAIT_TIME_TO_SCHEDULE = float(os.getenv("WAIT_TIME_TO_SCHEDULE", "1.0"))

WAIT_TIME_VAD = float(os.getenv("WAIT_TIME_VAD", "0.5"))

ASR_REQ_SIZE = int(os.getenv("ASR_REQ_SIZE", "32"))
WAIT_TIME_ASR = float(os.getenv("WAIT_TIME_ASR", "0.01"))

@concurrent.thread
def process_job(job_id, job_details):
    try:
        Logger.info(f"Processing Job: {job_id}")

        raw_audio_file = f"{RAW_AUDIO_PATH}/{job_id}.media"
        wav_file = f"{WAV_AUDIO_PATH}/{job_id}.wav"

        convert_to_wav16(raw_audio_file, wav_file)
        updateJobStatus(job_id, progress=5)

        vad.addTask(job_id, wav_file)
        updateJobStatus(job_id, progress=10)

        while vad.getResponse(job_id) is None:
            time.sleep(WAIT_TIME_VAD)

        start_ends = vad.getResponse(job_id)
        updateJobStatus(job_id, progress=20)

        audio_signal, audio_duration = load_audio(wav_file)
        segmented_audio_signal = get_segmented_audio_signal(start_ends, audio_signal)

        last_pbar = 0
        pbar_inc = 70/len(segmented_audio_signal)

        working_requests = []
        for current_req_idx in range(len(segmented_audio_signal)):
            if len(working_requests) == ASR_REQ_SIZE:
                while True:
                    finished_reqs = []
                    for req_id in working_requests:
                        if asr.getResponse(req_id) is not None:
                            finished_reqs.append(req_id)
                            last_pbar += pbar_inc
                            updateJobStatus(job_id, progress=last_pbar)

                    if len(finished_reqs):
                        for req_id in finished_reqs:
                            working_requests.remove(req_id)

                        break
                    
                    time.sleep(WAIT_TIME_ASR)

            job_data = {
                'audio': segmented_audio_signal[current_req_idx][0],
                **job_details
            }

            asr.addTask(f"{job_id}_{current_req_idx}", job_data)
            working_requests.append(f"{job_id}_{current_req_idx}")

        while len(working_requests) > 0:
            finished_reqs = []
            for req_id in working_requests:
                if asr.getResponse(req_id) is not None:
                    finished_reqs.append(req_id)
                    last_pbar += pbar_inc
                    updateJobStatus(job_id, progress=last_pbar)

            if len(finished_reqs):
                for req_id in finished_reqs:
                    working_requests.remove(req_id)
            
            time.sleep(WAIT_TIME_ASR)

        responses = [{**_aud[2], **asr.getResponse(f"{job_id}_{i}")} for i, _aud in enumerate(segmented_audio_signal)]
        finishTask(job_id, responses)

        for i in range(len(segmented_audio_signal)):
            asr.clearResponse(f"{job_id}_{i}")

        vad.clearResponse(job_id)

    except Exception as ex:
        failedTask(job_id, {'msg': f"{ex}", 'info': f"{traceback.format_exc()}"})


def start_worker():
    Logger.info(f"Starting worker!")

    working_requests = []
    while True:
        if len(working_requests) == MAX_CONCURRENT_REQ:
            finished_requests = []
            while True:
                for req in working_requests:
                    if req.done():
                        finished_requests.append(req)

                if len(finished_requests):
                    break
                else:
                    time.sleep(WAIT_TIME_TO_FINISH)

            for req in finished_requests:
                working_requests.remove(req)

        job_id, job_details = pullTask()
        if job_id is None:
            Logger.info(f"Waiting for new task...")
            time.sleep(WAIT_TIME_TO_SCHEDULE)
            continue
        else:
            working_requests.append(process_job(job_id, job_details))


if __name__ == "__main__":
    start_worker()