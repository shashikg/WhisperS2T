from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console

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