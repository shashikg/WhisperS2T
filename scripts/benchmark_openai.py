import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_path', default="", type=str)
    args = parser.parse_args()
    return args


def run(repo_path):
    import time, os
    from tqdm import tqdm
    import pandas as pd
    import whisper

    results_dir = f"{repo_path}/results/OpenAI"
    os.makedirs(results_dir, exist_ok=True)

    model = whisper.load_model('large-v2')
    model = model.cuda().eval()

    # KINCAID46 WAV >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    data = pd.read_csv(f'{repo_path}/data/KINCAID46/manifest_wav.tsv', sep="\t")
    files = [f"{repo_path}/{fn}" for fn in data['audio_path']]
    
    for fn in tqdm(files[:5], desc="warming"):
        result = model.transcribe(fn, language='en')

    st = time.time()
    pred_text = []
    for fn in tqdm(files, desc="KINCAID"):
        result = model.transcribe(fn, language='en')
        pred_text.append(result['text'].strip())
    
    time_kincaid46_wav = time.time()-st

    data['pred_text'] = pred_text
    data.to_csv(f"{results_dir}/KINCAID46_WAV.tsv", sep="\t", index=False)

    # MultiLingualLongform >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    data = pd.read_csv(f'{repo_path}/data/MultiLingualLongform/manifest.tsv', sep="\t")
    files = [f"{repo_path}/{fn}" for fn in data['audio_path']]
    lang_codes = data['lang_code'].to_list()

    st = time.time()
    pred_text = []
    for idx in tqdm(range(len(files)), desc="MultiLingualLongform"):
        result = model.transcribe(files[idx], language=lang_codes[idx])
        pred_text.append(result['text'].strip())
    
    time_multilingual = time.time()-st
    
    data['pred_text'] = pred_text
    data.to_csv(f"{results_dir}/MultiLingualLongform.tsv", sep="\t", index=False)

    infer_time = [
        ["Dataset", "Time"],
        ["KINCAID46 WAV", time_kincaid46_wav],
        ["KINCAID46 MP3", 0.0],
        ["MultiLingualLongform", time_multilingual]
    ]
    
    infer_time = pd.DataFrame(infer_time[1:], columns=infer_time[0])
    infer_time.to_csv(f"{results_dir}/infer_time.tsv", sep="\t", index=False)


if __name__ == '__main__':
    args = parse_arguments()
    run(args.repo_path)