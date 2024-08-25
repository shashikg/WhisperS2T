import os
import time
import argparse
import requests
import tempfile
import streamlit as st

from whisper_s2t.utils import write_outputs

st.set_page_config(
    page_title="WhisperS2T App",
    layout="wide",
    initial_sidebar_state="expanded"
)

LANGUAGES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "az": "Azerbaijani",
    "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan",
    "br": "Breton", "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "et": "Estonian", "eu": "Basque",
    "fa": "Persian", "fi": "Finnish", "fo": "Faroese", "fr": "French", "gl": "Galician", "gu": "Gujarati",
    "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole",
    "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "is": "Icelandic", "it": "Italian",
    "ja": "Japanese", "jw": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer", "kn": "Kannada",
    "ko": "Korean", "la": "Latin", "lb": "Luxembourgish", "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian",
    "lv": "Latvian", "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian",
    "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese", "ne": "Nepali", "nl": "Dutch",
    "nn": "Norwegian Nynorsk", "no": "Norwegian", "oc": "Occitan", "pa": "Punjabi", "pl": "Polish",
    "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit", "sd": "Sindhi",
    "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sn": "Shona", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "tg": "Tajik", "th": "Thai", "tk": "Turkmen", "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar",
    "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese", "yi": "Yiddish", "yo": "Yoruba",
    "zh": "Chinese"
}

TASKS = {
    "transcribe": "Transcribe",
    "translate": "Translate"
}

def get_transcript(file, lang, task, progress_bar, API_URL_STATUS, API_URL_TRANSCRIBE):
    """Upload the file to the transcription API and get the transcript with progress updates."""

    files = {'file': file}
    data = {'lang': lang, 'task': task}
    
    progress_bar.progress(0.0, text="Transcribing")
    
    response = requests.post(API_URL_TRANSCRIBE, files=files, data=data)
    
    # Extract job_id from the response
    job_id = response.json().get('job_id')
    
    # Continuously check status until completed or failed
    last_progress = 0.0
    while True:
        status_response = requests.get(API_URL_STATUS + job_id)
        status_data = status_response.json()
        
        status = status_data.get('status')
        progress = status_data.get('progress', last_progress)
        last_progress = progress

        # Update progress bar
        progress_bar.progress(min(progress/100.0, 1.0), text="Transcribing")
        
        if status in ["completed", "failed"]:
            progress_bar.progress(1.0, text="Transcribing")
            break
        
        time.sleep(1)  # Adjust the delay based on API's status update frequency

    return status_data

def convert_transcript(transcripts, format='txt'):
    with tempfile.TemporaryDirectory() as temp_dir:
        write_outputs([transcripts], format=format, op_files=[f"{temp_dir}/tmp.{format}"])

        with open(f"{temp_dir}/tmp.{format}", 'r') as f:
            out = f.read()

    return out

TranscriptExporter = {
    "TXT": 'txt',
    "VTT": 'vtt',
    "SRT": 'srt',
    "TSV": 'tsv',
    "JSON": 'json'
}

def display(show_pbar=True):
    if st.session_state.transcript_status.get('status') == "completed":
        if show_pbar:
            with st.sidebar:
                progress_bar = st.progress(1.0, text="Transcribing")

        with st.sidebar:
            st.success('Transcription completed!')

        transcripts = st.session_state.transcript_status.get('result', [])

        with st.sidebar:
            # Provide download options
            download_format = st.selectbox("Export format:", ["JSON", "VTT", "SRT", "TSV", "TXT"])
            st.download_button(label="Download", 
                            data=convert_transcript(transcripts, format=TranscriptExporter[download_format]), 
                            file_name=f"transcripts.{TranscriptExporter[download_format]}")
        
        with st.expander("TRANSCRIPTS:"):
            for line in convert_transcript(transcripts, format='txt').split("\n\n"):
                if line.strip():
                    st.info(line)
    else:
        st.error(f'Transcription failed with error: {st.session_state.transcript_status.get("error")}')

def main(server_port):
    API_URL_STATUS = f"http://localhost:{server_port}/status/"
    API_URL_TRANSCRIBE = f"http://localhost:{server_port}/transcribe"

    st.title("WhisperS2T App")

    uploaded_file = st.file_uploader("Upload Audio/Video File")

    if uploaded_file is not None:
        st.success("File uploaded successfully.")

    with st.sidebar:
        # Create a row for language and task selection
        lang_col, task_col = st.columns([1, 1])  # Adjust column widths as needed
        
        with lang_col:
            lang = st.selectbox("Language", 
                                options=list(LANGUAGES.keys()), 
                                format_func=lambda x: LANGUAGES[x], 
                                index=list(LANGUAGES.keys()).index("en"))
        
        with task_col:
            task = st.selectbox("Task", 
                                options=list(TASKS.keys()), 
                                format_func=lambda x: TASKS[x])

        transcribe_button = st.button("Transcribe")

    if transcribe_button:
        if uploaded_file is not None:
            st.session_state.last_file = uploaded_file

            with st.sidebar:
                progress_bar = st.progress(0.0, text="Transcribing")
                st.session_state.transcript_status = get_transcript(uploaded_file, lang, task, progress_bar, API_URL_STATUS, API_URL_TRANSCRIBE)

            display(show_pbar=False)
        else:
            st.error(f"Please upload the file first!")
    else:
        try:
            if st.session_state.last_file == uploaded_file:
                display()
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='whisper_s2t_server_app')
    parser.add_argument('--server_port', default=8000, type=int, help='server port')

    args = parser.parse_args()
    main(args.server_port)