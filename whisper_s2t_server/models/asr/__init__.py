import os
from diskcache import Index
from datetime import datetime

from ... import WHISPER_S2T_SERVER_TMP_PATH

ASR_REQ_PATH = f"{WHISPER_S2T_SERVER_TMP_PATH}/models_queue/asr/request"
ASR_RES_PATH = f"{WHISPER_S2T_SERVER_TMP_PATH}/models_queue/asr/response"

# os.system(f"rm -rf {ASR_REQ_PATH}")
# os.system(f"rm -rf {ASR_RES_PATH}")

os.makedirs(ASR_REQ_PATH, exist_ok=True)
os.makedirs(ASR_RES_PATH, exist_ok=True)

REQ_QUEUE = Index(ASR_REQ_PATH)
RES_QUEUE = Index(ASR_RES_PATH)

def addTask(job_id, job_data):
    REQ_QUEUE[job_id] = job_data

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