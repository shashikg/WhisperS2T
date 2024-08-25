import logging
from rich.logging import RichHandler


LOG_LEVEL = "INFO"

formatter = logging.Formatter("%(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
rich_handler = RichHandler(rich_tracebacks=True)
rich_handler.setFormatter(formatter)

Logger = logging.getLogger("whisper_s2t_server")
Logger.addHandler(rich_handler)
Logger.setLevel(LOG_LEVEL)