from gevent import monkey
monkey.patch_all()

import uuid
import falcon
from falcon_multipart.middleware import MultipartMiddleware

from . import RAW_AUDIO_PATH
from .logger import Logger
from .utils import getJobStatus, addTask

class TranscribeAPI:
    def on_post(self, req, resp):
        audio_media = req.get_param('file')
        lang = req.get_param('lang')
        task = req.get_param('task')
        initial_prompt = req.get_param('prompt')
        
        if lang is None:
            resp.status = falcon.HTTP_400
            resp.media = {'error': 'Missing required parameter: `lang`'}
            return
        
        if task is None:
            resp.status = falcon.HTTP_400
            resp.media = {'error': 'Missing required parameter: `task`'}
            return

        if audio_media is None:
            resp.status = falcon.HTTP_400
            resp.media = {'error': 'Missing required audio/video media file.'}
            return

        job_id = str(uuid.uuid4())
        file_path = f"{RAW_AUDIO_PATH}/{job_id}.media"

        with open(file_path, 'wb') as f:
            while True:
                chunk = audio_media.file.read(4096)
                if not chunk:
                    break

                f.write(chunk)

        addTask(job_id, lang, task, initial_prompt)

        resp.media = {
            'message': f'Request received!',
            'job_id': job_id
        }

        resp.status = falcon.HTTP_201


class JobStatusAPI:
    def on_get(self, req, resp, job_id):
        job_status = getJobStatus(job_id)
        
        if job_status:
            resp.status = falcon.HTTP_201
            resp.media = job_status
        else:
            resp.status = falcon.HTTP_404
            resp.media = {'error': f'Job ID: {job_id} not found!'}


app = falcon.App(middleware=[MultipartMiddleware()])
app.add_route('/transcribe', TranscribeAPI())
app.add_route('/status/{job_id}', JobStatusAPI())