import logging
from rich.logging import RichHandler

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


LOG_LEVEL = "INFO"

formatter = logging.Formatter("%(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
rich_handler = RichHandler(rich_tracebacks=True)
rich_handler.setFormatter(formatter)

Logger = logging.getLogger("whisper_s2t_server")
Logger.addHandler(rich_handler)
Logger.setLevel(LOG_LEVEL)


class LogFileHandler(FileSystemEventHandler):
    def __init__(self, file_paths):
        self.file_pointers = {}
        # Open the log files and move the pointer to the end
        for path in file_paths:
            try:
                f = open(path, 'r')
                # Read the entire file content initially
                content = f.read()
                if content:
                    print(content, end='')

                # Move the pointer to the end of the file for live monitoring
                f.seek(0, 2)
                self.file_pointers[path] = f
            except Exception as e:
                print(f"Error opening {path}: {e}")

    def on_modified(self, event):
        # When a file is modified, read the new content
        if event.src_path in self.file_pointers:
            f = self.file_pointers[event.src_path]
            new_lines = f.readlines()
            if new_lines:
                for line in new_lines:
                    print(f"{line}", end='')

    def close_files(self):
        for f in self.file_pointers.values():
            f.close()


def StreamLogs(log_files):
    # Create a handler for the log files
    event_handler = LogFileHandler(log_files)

    # Set up an observer to watch the directory containing the logs
    observer = Observer()

    # You can add a separate watch for each directory if the logs are in different directories
    for log_file in log_files:
        observer.schedule(event_handler, path=log_file, recursive=False)

    # Start the observer
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped monitoring.")
        event_handler.close_files()
        observer.stop()
        
    observer.join()