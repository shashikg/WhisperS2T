import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_path', default="", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()
    return args


def run(repo_path, batch_size=16):
    import time, os
    import whisperx
    from tqdm import tqdm
    import pandas as pd

    results_dir = f"{repo_path}/results/WhisperX-bs_{batch_size}"
    os.makedirs(results_dir, exist_ok=True)

    model = whisperx.load_model("large-v2", "cuda", compute_type="float16", language='en', asr_options={'beam_size': 1})

    # KINCAID46 WAV >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    data = pd.read_csv(f'{repo_path}/data/KINCAID46/manifest_wav.tsv', sep="\t")
    files = [f"{repo_path}/{fn}" for fn in data['audio_path']]
    
    for fn in tqdm(files, desc="Warming"):
        audio = whisperx.load_audio(fn)
        result = model.transcribe(audio, batch_size=batch_size)

    st = time.time()
    pred_text = []
    for fn in tqdm(files, desc="KINCAID WAV"):
        audio = whisperx.load_audio(fn)
        result = model.transcribe(audio, batch_size=batch_size)
        pred_text.append(" ".join([_['text'].strip() for _ in result['segments']]))
    
    time_kincaid46_wav = time.time()-st

    data['pred_text'] = pred_text
    data.to_csv(f"{results_dir}/KINCAID46_WAV.tsv", sep="\t", index=False)


    # KINCAID46 MP3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    data = pd.read_csv(f'{repo_path}/data/KINCAID46/manifest_mp3.tsv', sep="\t")
    files = [f"{repo_path}/{fn}" for fn in data['audio_path']]

    st = time.time()
    pred_text = []
    for fn in tqdm(files, desc="KINCAID MP3"):
        audio = whisperx.load_audio(fn)
        result = model.transcribe(audio, batch_size=batch_size)
        pred_text.append(" ".join([_['text'].strip() for _ in result['segments']]))
    
    time_kincaid46_mp3 = time.time()-st

    data['pred_text'] = pred_text
    data.to_csv(f"{results_dir}/KINCAID46_MP3.tsv", sep="\t", index=False)


    # MultiLingualLongform >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    data = pd.read_csv(f'{repo_path}/data/MultiLingualLongform/manifest.tsv', sep="\t")
    files = [f"{repo_path}/{fn}" for fn in data['audio_path']]
    lang_codes = data['lang_code'].to_list()

    st = time.time()
    pred_text = []
    for idx in tqdm(range(len(files)), desc="MultiLingualLongform"):
        audio = whisperx.load_audio(files[idx])
        result = model.transcribe(audio, batch_size=batch_size, language=lang_codes[idx], task='transcribe')
        pred_text.append(" ".join([_['text'].strip() for _ in result['segments']]))
    
    time_multilingual = time.time()-st
    
    data['pred_text'] = pred_text
    data.to_csv(f"{results_dir}/MultiLingualLongform.tsv", sep="\t", index=False)

    infer_time = [
        ["Dataset", "Time"],
        ["KINCAID46 WAV", time_kincaid46_wav],
        ["KINCAID46 MP3", time_kincaid46_mp3],
        ["MultiLingualLongform", time_multilingual]
    ]
    infer_time = pd.DataFrame(infer_time[1:], columns=infer_time[0])
    infer_time.to_csv(f"{results_dir}/infer_time.tsv", sep="\t", index=False)


if __name__ == '__main__':
    args = parse_arguments()
    run(args.repo_path, batch_size=args.batch_size)