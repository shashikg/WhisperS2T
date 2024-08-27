import os
import shutil
import signal
import argparse
import subprocess

from . import WHISPER_S2T_SERVER_TMP_PATH, BASE_PATH
from .logger import LogFileHandler, StreamLogs


def run_subprocess(cmd, log_file):
    return subprocess.Popen(" ".join(cmd) + f" > {log_file} 2>&1", shell=True)
    
def start_server(server_port, asr_args, vad_args, app_args):
    # shutil.rmtree(f"{WHISPER_S2T_SERVER_TMP_PATH}/")
    os.makedirs(f'{WHISPER_S2T_SERVER_TMP_PATH}/logs', exist_ok=True)
    
    # Start the Gunicorn server
    rest_server_command = [
        "gunicorn",
        "--preload",
        "-t", "180",
        "--worker-class=gevent",
        "--worker-connections=1000",
        "--workers=3",
        "--threads=3",
        "-b", f"0.0.0.0:{server_port}",
        "whisper_s2t_server.server:app"
    ]
    rest_server_process = run_subprocess(rest_server_command, f'{WHISPER_S2T_SERVER_TMP_PATH}/logs/rest_server.log')

    # with open(f'{WHISPER_S2T_SERVER_TMP_PATH}/logs/rest_server.log', 'a') as log_file:
    #     rest_server_process = subprocess.Popen(rest_server_command, stdout=log_file, stderr=subprocess.STDOUT)
    
    # Start the ASR process
    asr_input_args = [f"--{k}={v}" for k, v in asr_args.items()]
    asr_command = ["python3", "-m", "whisper_s2t_server.models.asr.main"] + asr_input_args
    asr_process = run_subprocess(asr_command, f'{WHISPER_S2T_SERVER_TMP_PATH}/logs/asr.log')

    # with open(f'{WHISPER_S2T_SERVER_TMP_PATH}/logs/asr.log', 'a') as log_file:
    #     asr_process = subprocess.Popen(asr_command, stdout=log_file, stderr=subprocess.STDOUT)
    
    # Start the VAD process
    vad_input_args = [f"--{k}={v}" for k, v in vad_args.items()]
    vad_command = ["python3", "-m", "whisper_s2t_server.models.vad.main"] + vad_input_args
    vad_process = run_subprocess(vad_command, f'{WHISPER_S2T_SERVER_TMP_PATH}/logs/vad.log')

    # with open(f'{WHISPER_S2T_SERVER_TMP_PATH}/logs/vad.log', 'a') as log_file:
    #     vad_process = subprocess.Popen(vad_command, stdout=log_file, stderr=subprocess.STDOUT)

    # Start Main Worker
    main_worker_command = ["python3", "-m", "whisper_s2t_server.worker"]
    main_worker_process = run_subprocess(main_worker_command, f'{WHISPER_S2T_SERVER_TMP_PATH}/logs/main_worker.log')

    # with open(f'{WHISPER_S2T_SERVER_TMP_PATH}/logs/main_worker.log', 'a') as log_file:
    #     main_worker_process = subprocess.Popen(main_worker_command, stdout=log_file, stderr=subprocess.STDOUT)

    # Start the Streamlit App process
    app_input_args = [
        f"--server.maxUploadSize=1024",
        f"--server.port={app_args['app_server_port']}", 
        f"--",
        f"--server_port={app_args['server_port']}"
    ]
    app_command = ["streamlit", "run", f"{BASE_PATH}/app.py"] + app_input_args
    app_process = run_subprocess(app_command, f'{WHISPER_S2T_SERVER_TMP_PATH}/logs/app.log')

    # with open(f'{WHISPER_S2T_SERVER_TMP_PATH}/logs/app.log', 'a') as log_file:
    #     app_process = subprocess.Popen(app_command, stdout=log_file, stderr=subprocess.STDOUT)
    
    # Save PIDs to a file
    with open(f'{WHISPER_S2T_SERVER_TMP_PATH}/logs/process_pids.txt', 'w') as pid_file:
        pid_file.write(f"rest_server:{rest_server_process.pid}\n")
        pid_file.write(f"asr:{asr_process.pid}\n")
        pid_file.write(f"vad:{vad_process.pid}\n")
        pid_file.write(f"app:{app_process.pid}\n")
        pid_file.write(f"main_worker:{main_worker_process.pid}\n")

def view_logs():

    log_files = [
        f"{WHISPER_S2T_SERVER_TMP_PATH}/logs/app.log",
        f"{WHISPER_S2T_SERVER_TMP_PATH}/logs/asr.log",
        f"{WHISPER_S2T_SERVER_TMP_PATH}/logs/vad.log",
        f"{WHISPER_S2T_SERVER_TMP_PATH}/logs/rest_server.log",
        f"{WHISPER_S2T_SERVER_TMP_PATH}/logs/main_worker.log"
    ]

    StreamLogs(log_files)


def stop_server():
    try:
        with open(f'{WHISPER_S2T_SERVER_TMP_PATH}/logs/process_pids.txt', 'r') as pid_file:
            lines = pid_file.readlines()
            for line in lines:
                try:
                    process_name, pid = line.strip().split(':')
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"Killed {process_name} with PID: {pid}")
                except Exception as ex:
                    print(f"Failed to kill {process_name} with PID: {pid}. Error: {ex}")
                    pass
        
        os.remove(f'{WHISPER_S2T_SERVER_TMP_PATH}/logs/process_pids.txt')
    except FileNotFoundError:
        print("No running server found.")
    except ProcessLookupError:
        print("Server process not found. It might already be stopped.")

def main():
    parser = argparse.ArgumentParser(prog='whisper_s2t_server')
    subparsers = parser.add_subparsers(dest='command')

    start_parser = subparsers.add_parser('start', help='Start the server')
    start_parser.add_argument('--server_port', default=8000, type=int, help='server port')
    start_parser.add_argument('--app_server_port', default=8001, type=int, help='app server port')
    start_parser.add_argument('--model_identifier', default="tiny", help='Model name to use')
    start_parser.add_argument('--backend', default='ct2', help='Which backend to use')
    start_parser.add_argument('--device', default='cpu', help='cpu/cuda')

    subparsers.add_parser('logs', help='View server logs')
    subparsers.add_parser('stop', help='Stop the server')

    args = parser.parse_args()
    
    if args.command == 'start':
        asr_args = {
            "model_identifier": args.model_identifier,
            "backend": args.backend,
            "device": args.device,
        }

        vad_args = {
            "device": args.device,
        }

        app_args = {
            "server_port": args.server_port,
            "app_server_port": args.app_server_port,
        }

        start_server(args.server_port, asr_args, vad_args, app_args)
    elif args.command == 'logs':
        view_logs()
    elif args.command == 'stop':
        stop_server()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()