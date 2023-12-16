import os
from functools import cached_property

from ... import BASE_PATH


_TASKS = (
    "transcribe",
    "translate",
)


with open(os.path.join(BASE_PATH, "assets/lang_codes.txt"), 'r') as f:
    _LANGUAGE_CODES = [_ for _ in f.read().split("\n") if _]


class Tokenizer:
    def __init__(self, tokenizer, multilingual):
        
        self.tokenizer = tokenizer
        self.multilingual = multilingual
        
        if self.multilingual:
            self.task_to_token_id = {task: self.tokenizer.token_to_id(f"<|{task}|>") for task in _TASKS}
            self.lang_code_to_token_id = {lang: self.tokenizer.token_to_id(f"<|{lang}|>") for lang in _LANGUAGE_CODES}
        else:
            self.task_to_token_id = None
            self.lang_code_to_token_id = None

    @cached_property
    def transcribe(self) -> int:
        return self.tokenizer.token_to_id("<|transcribe|>")

    @cached_property
    def translate(self) -> int:
        return self.tokenizer.token_to_id("<|translate|>")
    
    @cached_property
    def silent_token(self) -> int:
        return self.encode(" ")[0]

    @cached_property
    def sot(self) -> int:
        return self.tokenizer.token_to_id("<|startoftranscript|>")

    @cached_property
    def sot_lm(self) -> int:
        return self.tokenizer.token_to_id("<|startoflm|>")

    @cached_property
    def sot_prev(self) -> int:
        return self.tokenizer.token_to_id("<|startofprev|>")

    @cached_property
    def eot(self) -> int:
        return self.tokenizer.token_to_id("<|endoftext|>")

    @cached_property
    def no_timestamps(self) -> int:
        return self.tokenizer.token_to_id("<|notimestamps|>")

    @property
    def timestamp_begin(self) -> int:
        return self.no_timestamps + 1

    def sot_sequence(self, task=None, lang=None):
        sequence = [self.sot]
        
        if self.multilingual:
            sequence.append(self.lang_code_to_token_id[lang])
            sequence.append(self.task_to_token_id[task])

        return sequence

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens):
        text_tokens = [token for token in tokens if token < self.eot]
        return self.tokenizer.decode(text_tokens)
    
    def decode_batch(self, tokens):
        res = []
        for tk in tokens:
            res.append([token for token in tk if token < self.eot])

        return self.tokenizer.decode_batch(res)