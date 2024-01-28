import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_path', default="", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--eval_mp3', default="yes", type=str)
    parser.add_argument('--eval_multilingual', default="yes", type=str)
    args = parser.parse_args()
    return args

def run(repo_path, batch_size=16, eval_mp3=True, eval_multilingual=True):
    import sys, time, os

    if len(repo_path):
        sys.path.append(repo_path)

    import whisper_s2t
    from whisper_s2t.backends.tensorrt.engine_builder import TRTBuilderConfig

    import pandas as pd

    results_dir = f"{repo_path}/results/WhisperS2T-TensorRT-LLM-bs_{batch_size}"
    os.makedirs(results_dir, exist_ok=True)

    trt_build_args = TRTBuilderConfig(
        max_batch_size=batch_size, 
        max_output_len=448
    )

    model_kwargs = {
        'trt_build_args': trt_build_args,
    }

    model = whisper_s2t.load_model(model_identifier="large-v2", backend='TensorRT-LLM', **model_kwargs)

    # KINCAID46 WAV >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    data = pd.read_csv(f'{repo_path}/data/KINCAID46/manifest_wav.tsv', sep="\t")
    files = [f"{repo_path}/{fn}" for fn in data['audio_path']]
    lang_codes = len(files)*['en']
    tasks = len(files)*['transcribe']
    initial_prompts = len(files)*[None]
    
    _ = model.transcribe_with_vad(files,
                                    lang_codes=lang_codes, 
                                    tasks=tasks, 
                                    initial_prompts=initial_prompts,
                                    batch_size=batch_size)

    st = time.time()
    out = model.transcribe_with_vad(files,
                                    lang_codes=lang_codes, 
                                    tasks=tasks, 
                                    initial_prompts=initial_prompts,
                                    batch_size=batch_size)
    time_kincaid46_wav = time.time()-st
    
    data['pred_text'] = [" ".join([_['text'] for _ in _transcript]).strip() for _transcript in out]
    data.to_csv(f"{results_dir}/KINCAID46_WAV.tsv", sep="\t", index=False)


    # KINCAID46 MP3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if eval_mp3:
        data = pd.read_csv(f'{repo_path}/data/KINCAID46/manifest_mp3.tsv', sep="\t")
        files = [f"{repo_path}/{fn}" for fn in data['audio_path']]
        lang_codes = len(files)*['en']
        tasks = len(files)*['transcribe']
        initial_prompts = len(files)*[None]

        st = time.time()
        out = model.transcribe_with_vad(files,
                                        lang_codes=lang_codes, 
                                        tasks=tasks, 
                                        initial_prompts=initial_prompts,
                                        batch_size=batch_size)
        time_kincaid46_mp3 = time.time()-st
        
        data['pred_text'] = [" ".join([_['text'] for _ in _transcript]).strip() for _transcript in out]
        data.to_csv(f"{results_dir}/KINCAID46_MP3.tsv", sep="\t", index=False)
    else:
        time_kincaid46_mp3 = 0.0


    # MultiLingualLongform
    if eval_multilingual:
        data = pd.read_csv(f'{repo_path}/data/MultiLingualLongform/manifest.tsv', sep="\t")
        files = [f"{repo_path}/{fn}" for fn in data['audio_path']]
        lang_codes = data['lang_code'].to_list()
        tasks = len(files)*['transcribe']
        initial_prompts = len(files)*[None]

        st = time.time()
        out = model.transcribe_with_vad(files,
                                        lang_codes=lang_codes, 
                                        tasks=tasks, 
                                        initial_prompts=initial_prompts,
                                        batch_size=batch_size)
        time_multilingual = time.time()-st
        
        data['pred_text'] = [" ".join([_['text'] for _ in _transcript]).strip() for _transcript in out]
        data.to_csv(f"{results_dir}/MultiLingualLongform.tsv", sep="\t", index=False)
    else:
        time_multilingual = 0.0

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
    eval_mp3 = True if args.eval_mp3 == "yes" else False
    eval_multilingual = True if args.eval_multilingual == "yes" else False
    run(args.repo_path, batch_size=args.batch_size, eval_mp3=eval_mp3, eval_multilingual=eval_multilingual)