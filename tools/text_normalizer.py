import os
import re
import json
import contractions
from cleantext import clean
from nemo_text_processing.text_normalization.normalize import Normalizer


def BlankFunction(text):
    return text

class EnglishSpellingNormalizer:
    """
    [Note]: Taken from OpenAI Whisper repo: https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py#L450
    
    Applies British-American spelling mappings as listed in [1].

    [1] https://www.tysto.com/uk-us-spelling-list.html
    """

    def __init__(self):
        mapping_path = os.path.join(os.path.dirname(__file__), "english.json")
        self.mapping = json.load(open(mapping_path))

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())


filler_words = ["um", "uh", "uhuh", "mhm", "ah", "mm", "mmm", "mn", "hmm", "hm", "huh", "heh", "eh", "eah", "ha", "aw", "ye", "ey", "aha", "ugh"]
filler_words = {k: False for k in filler_words}
def clean_text(text, replacers={}, remove_filler_words=True):
    text = text.lower()
    
    for pattern, replacement in replacers.items():
        text = re.sub(pattern, replacement, text)
            
    text = contractions.fix(text)
    
    text = re.sub('(\.)(com|org|net|ai|su)', r' dot \2', text) # fix websites
    text = re.sub('(www\.)([a-z])', r'www dot \2', text) # fix websites
    text = re.sub('-|~|\[.+?\]|\(.+?\)|\{.+?\}|\$|\^|\+|\=|\>|\<|\|', ' ', text) # remove anything insider brackets [..] (..) 

    text = re.sub('([a-z]/)([a-z])', r'\1 or \2', text)
    text = text.replace("½", "half")
    text = text.replace("¼", "quarter")
    text = text.replace(" ok ", " okay ")
            
    text = re.sub("\u2019|'", "APSTROPH", text)
    
    text = clean(text,
                 fix_unicode=False,
                 to_ascii=False,
                 lower=False,
                 no_line_breaks=True,
                 no_punct=True,
                 no_urls=False,
                 no_emails=False,
                 no_phone_numbers=False,
                 no_numbers=False,
                 no_digits=False,
                 no_currency_symbols=False)
    
    if remove_filler_words:
        words = []
        for w in text.split(" "):
            if filler_words.get(w, True) and len(w.strip()):
                words.append(w)

        text = " ".join(words)
    
    text = text.replace("APSTROPH", "'")
    text = text.lower()
    return text

class TextNormalizer:
    def __init__(self, lang='en', remove_filler_words=True):
        """
        [Note]: Taken from OpenAI Whisper repo: https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py#L465
        """

        self.standardize_spellings = BlankFunction  
        self.normalizer = Normalizer(lang=lang, input_case='cased')
        self.remove_filler_words = remove_filler_words
        
        self.replacers = {}
        if lang=='en':
            self.standardize_spellings = EnglishSpellingNormalizer()
            self.replacers = {
                # common contractions
                r"\bwon't\b": "will not",
                r"\bcan't\b": "can not",
                r"\blet's\b": "let us",
                r"\bain't\b": "aint",
                r"\by'all\b": "you all",
                r"\bwanna\b": "want to",
                r"\bgotta\b": "got to",
                r"\bgonna\b": "going to",
                r"\bi'ma\b": "i am going to",
                r"\bimma\b": "i am going to",
                r"\bwoulda\b": "would have",
                r"\bcoulda\b": "could have",
                r"\bshoulda\b": "should have",
                r"\bma'am\b": "madam",
                # contractions in titles/prefixes
                r"\bmr\b": "mister ",
                r"\bmrs\b": "missus ",
                r"\bst\b": "saint ",
                r"\bdr\b": "doctor ",
                r"\bprof\b": "professor ",
                r"\bcapt\b": "captain ",
                r"\bgov\b": "governor ",
                r"\bald\b": "alderman ",
                r"\bgen\b": "general ",
                r"\bsen\b": "senator ",
                r"\brep\b": "representative ",
                r"\bpres\b": "president ",
                r"\brev\b": "reverend ",
                r"\bhon\b": "honorable ",
                r"\basst\b": "assistant ",
                r"\bassoc\b": "associate ",
                r"\blt\b": "lieutenant ",
                r"\bcol\b": "colonel ",
                r"\bjr\b": "junior ",
                r"\bsr\b": "senior ",
                r"\besq\b": "esquire ",
                # prefect tenses, ideally it should be any past participles, but it's harder..
                r"'d been\b": " had been",
                r"'s been\b": " has been",
                r"'d gone\b": " had gone",
                r"'s gone\b": " has gone",
                r"'d done\b": " had done",  # "'s done" is ambiguous
                r"'s got\b": " has got",
                # general contractions
                r"n't\b": " not",
                r"'re\b": " are",
                r"'s\b": " is",
                r"'d\b": " would",
                r"'ll\b": " will",
                r"'t\b": " not",
                r"'ve\b": " have",
                r"'m\b": " am",
            }
        
    def __call__(self, txt):
        norm_txt = []
        for sent in txt.split(". "):
            norm_sent = []
            for sub_sent in sent.split(", "):
                if len(re.sub('[0123456789]', '', sub_sent)) != len(sub_sent):
                    sub_sent = self.normalizer.normalize(sub_sent, verbose=False).strip()
                    
                if len(sub_sent):
                    norm_sent.append(sub_sent.strip())
                    
            norm_txt.append(", ".join(norm_sent))
        
        norm_txt = [self.standardize_spellings(clean_text(_, replacers=self.replacers, remove_filler_words=self.remove_filler_words)) for _ in norm_txt]
        
        return " ".join(norm_txt)