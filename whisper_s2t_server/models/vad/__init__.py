import os
from diskcache import Index
from datetime import datetime

from ... import WHISPER_S2T_SERVER_TMP_PATH

VAD_REQ_PATH = f"{WHISPER_S2T_SERVER_TMP_PATH}/models_queue/vad/request"
VAD_RES_PATH = f"{WHISPER_S2T_SERVER_TMP_PATH}/models_queue/vad/response"

# os.system(f"rm -rf {VAD_REQ_PATH}")
# os.system(f"rm -rf {VAD_RES_PATH}")

os.makedirs(VAD_REQ_PATH, exist_ok=True)
os.makedirs(VAD_RES_PATH, exist_ok=True)

REQ_QUEUE = Index(VAD_REQ_PATH)
RES_QUEUE = Index(VAD_RES_PATH)

def addTask(job_id, input_file):
    REQ_QUEUE[job_id] = input_file

def pullTask():
    try:
        return REQ_QUEUE.popitem(last=False)
    except KeyError:
        return None, None
    
def finishTask(job_id, result):
    RES_QUEUE[job_id] = result

def getResponse(job_id):
    return RES_QUEUE.get(job_id, None)

def clearResponse(job_id):
    del RES_QUEUE[job_id]