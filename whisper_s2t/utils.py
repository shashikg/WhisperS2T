import os
import json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn


class RunningStatus:
    def __init__(self, status_text, console=None):
        self.status_text = status_text
        if console:
            self.console = console
        else:
            self.console = Console()

    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False
        )
        self.task = self.progress.add_task(f"{self.status_text}", total=None)
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.update(self.task, advance=1.0)  # Complete the progress bar
        self.progress.stop()  # Stop the progress display


def format_timestamp(seconds, always_include_hours=False, decimal_marker="."):

    assert seconds >= 0, "non-negative timestamp expected"

    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def ExportVTT(transcript, file):

    with open(file, 'w') as f:
        f.write("WEBVTT\n\n")
        for _utt in transcript:
            f.write(f"{format_timestamp(_utt['start_time'])} --> {format_timestamp(_utt['end_time'])}\n{_utt['text']}\n\n")


def ExportSRT(transcript, file):

    with open(file, 'w') as f:
        for i, _utt in enumerate(transcript):
            
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(_utt['start_time'], always_include_hours=True, decimal_marker=',')} --> ")
            f.write(f"{format_timestamp(_utt['end_time'], always_include_hours=True, decimal_marker=',')}\n")
            f.write(f"{_utt['text']}\n\n")


def ExportJSON(transcript, file):

    with open(file, 'w') as f:
        f.write(json.dumps(transcript))


def ExportTSV(transcript, file):
    keys = ['start_time', 'end_time', 'text']
    if len(transcript):
        for k in transcript[0].keys():
            if k not in keys: keys.append(k)

    with open(file, 'w') as f:
        f.write("\t".join(keys))
        for _utt in transcript:
            f.write("\n" + "\t".join([_utt[k].strip().replace("\t", " ") if k == 'text' else str(_utt[k]) for k in keys]))


TranscriptExporter = {
    'vtt': ExportVTT,
    'srt': ExportSRT,
    'tsv': ExportTSV,
    'json': ExportJSON
}


def write_outputs(transcripts, format='json', ip_files=None, op_files=None, save_dir="./"):
    if (op_files is None) or (len(op_files) != len(transcripts)):
        os.makedirs(save_dir, exist_ok=True)

        op_files = []

        if (ip_files is None) or (len(ip_files) != len(transcripts)):
            for i in range(len(transcripts)):
                op_files.append(os.path.join(save_dir, f"{i}.{format}"))
        else:
            for i, _ip_fn in enumerate(ip_files):
                base_name = ".".join(os.path.basename(_ip_fn).split(".")[:-1])
                op_files.append(os.path.join(save_dir, f"{i}_{base_name}.{format}"))

    
    for transcript, file_name in zip(transcripts, op_files):
        TranscriptExporter[format](transcript, file_name)