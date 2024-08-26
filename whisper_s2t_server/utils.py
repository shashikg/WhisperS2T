import os
from diskcache import Index
from datetime import datetime

from . import WHISPER_S2T_SERVER_TMP_PATH, RAW_AUDIO_PATH, WAV_AUDIO_PATH
from .logger import Logger

STATUS_PATH = f"{WHISPER_S2T_SERVER_TMP_PATH}/job/status"
REQ_PATH = f"{WHISPER_S2T_SERVER_TMP_PATH}/job/request"
RES_PATH = f"{WHISPER_S2T_SERVER_TMP_PATH}/job/response"

# os.system(f"rm -rf {STATUS_PATH}")
# os.system(f"rm -rf {REQ_PATH}")
# os.system(f"rm -rf {RES_PATH}")

os.makedirs(STATUS_PATH, exist_ok=True)
os.makedirs(REQ_PATH, exist_ok=True)
os.makedirs(RES_PATH, exist_ok=True)

STATUS_DB = Index(STATUS_PATH)
REQ_QUEUE = Index(REQ_PATH)
RES_QUEUE = Index(REQ_PATH)


def updateJobStatus(job_id, new_job=False, **kwargs):
    
    prev_status = STATUS_DB.get(job_id, {})
    if new_job:
        prev_status = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "received_at": f"{datetime.now()}",
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }
    else:
        prev_status.update(kwargs)

    STATUS_DB[job_id] = prev_status


def getJobStatus(job_id):
    return STATUS_DB.get(job_id, {})


def addTask(job_id, lang, task, initial_prompt):
    REQ_QUEUE[job_id] = {
        'lang': lang,
        'task': task,
        'initial_prompt': initial_prompt,
    }
    Logger.info(f"Added new task with job id: {job_id}")

    updateJobStatus(job_id, new_job=True)


def pullTask():
    try:
        job_id, job_details = REQ_QUEUE.popitem(last=False)
        updateJobStatus(job_id, status="in_progress", started_at=f"{datetime.now()}")
        Logger.info(f"Pulled new task with job id: {job_id}")
        return job_id, job_details
    except KeyError:
        return None, None
    

def finishTask(job_id, result):
    updateJobStatus(job_id, 
                    status="completed", 
                    progress=100,
                    completed_at=f"{datetime.now()}", 
                    result=result)
    
    if os.path.exists(f"{RAW_AUDIO_PATH}/{job_id}.media"):
        os.remove(f"{RAW_AUDIO_PATH}/{job_id}.media")

    if os.path.exists(f"{WAV_AUDIO_PATH}/{job_id}.wav"):
        os.remove(f"{WAV_AUDIO_PATH}/{job_id}.wav")


def failedTask(job_id, error):
    updateJobStatus(job_id, 
                    status="failed", 
                    progress=100, 
                    completed_at=f"{datetime.now()}", 
                    error=error)
    
    if os.path.exists(f"{RAW_AUDIO_PATH}/{job_id}.media"):
        os.remove(f"{RAW_AUDIO_PATH}/{job_id}.media")

    if os.path.exists(f"{WAV_AUDIO_PATH}/{job_id}.wav"):
        os.remove(f"{WAV_AUDIO_PATH}/{job_id}.wav")