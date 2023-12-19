import argparse
from rich.console import Console
console = Console()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_path', default="", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--flash_attention', default="yes", type=str)
    parser.add_argument('--better_transformer', default="no", type=str)
    parser.add_argument('--eval_mp3', default="no", type=str)
    parser.add_argument('--eval_multilingual', default="no", type=str)
    args = parser.parse_args()
    return args


def run(repo_path, flash_attention=False, better_transformer=False, batch_size=16, eval_mp3=False, eval_multilingual=True):
    import torch
    import time, os
    import pandas as pd
    from transformers import pipeline

    # Load Model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    model_kwargs = {
        "use_safetensors": True,
        "low_cpu_mem_usage": True
    }

    results_dir = f"{repo_path}/results/HuggingFaceDistilWhisper-bs_{batch_size}"

    if flash_attention:
        results_dir = f"{results_dir}-fa"
        model_kwargs["use_flash_attention_2"] = True

    ASR = pipeline("automatic-speech-recognition",
                   f"distil-whisper/distil-large-v2",
                   num_workers=1,
                   torch_dtype=torch.float16,
                   device="cuda",
                   model_kwargs=model_kwargs)
    
    if (not flash_attention) and better_transformer:
        ASR.model = ASR.model.to_bettertransformer()
        results_dir = f"{results_dir}-bt"
    
    os.makedirs(results_dir, exist_ok=True)

    # KINCAID46 WAV >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    data = pd.read_csv(f'{repo_path}/data/KINCAID46/manifest_wav.tsv', sep="\t")
    files = [f"{repo_path}/{fn}" for fn in data['audio_path']]
    
    with console.status("Warming"):
        st = time.time()
        _ = ASR(files,
                batch_size=batch_size,
                chunk_length_s=15,
                generate_kwargs={'num_beams': 1, 'language': 'en'},
                return_timestamps=False)

        print(f"[Warming Time]: {time.time()-st}")
    
    with console.status("KINCAID WAV"):
        st = time.time()
        outputs = ASR(files,
                       batch_size=batch_size,
                       chunk_length_s=15,
                       generate_kwargs={'num_beams': 1, 'language': 'en'},
                       return_timestamps=False)

        time_kincaid46_wav = time.time()-st
        print(f"[KINCAID WAV Time]: {time_kincaid46_wav}")
    
    data['pred_text'] = [_['text'].strip() for _ in outputs]
    data.to_csv(f"{results_dir}/KINCAID46_WAV.tsv", sep="\t", index=False)


    # KINCAID46 MP3 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if eval_mp3:
        data = pd.read_csv(f'{repo_path}/data/KINCAID46/manifest_mp3.tsv', sep="\t")
        files = [f"{repo_path}/{fn}" for fn in data['audio_path']]

        with console.status("KINCAID MP3"):
            st = time.time()
            outputs = ASR(files,
                        batch_size=batch_size,
                        chunk_length_s=30,
                        generate_kwargs={'num_beams': 1, 'language': 'en'},
                        return_timestamps=False)

            time_kincaid46_mp3 = time.time()-st

            print(f"[KINCAID MP3 Time]: {time_kincaid46_mp3}")
        
        data['pred_text']  = [_['text'].strip() for _ in outputs]
        data.to_csv(f"{results_dir}/KINCAID46_MP3.tsv", sep="\t", index=False)
    else:
        time_kincaid46_mp3 = 0.0

    # MultiLingualLongform >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if eval_multilingual:
        data = pd.read_csv(f'{repo_path}/data/MultiLingualLongform/manifest.tsv', sep="\t")
        files = [f"{repo_path}/{fn}" for fn in data['audio_path']]
        lang_codes = data['lang_code'].to_list()
        
        with console.status("MultiLingualLongform"):
            st = time.time()

            curr_files = [files[0]]
            curr_lang = lang_codes[0]
            outputs = []
            for fn, lang in zip(files[1:], lang_codes[1:]): 
                if lang != curr_lang:
                    _outputs = ASR(curr_files,
                                    batch_size=batch_size,
                                    chunk_length_s=30,
                                    generate_kwargs={'num_beams': 1, 'language': curr_lang},
                                    return_timestamps=False)
                    outputs.extend(_outputs)

                    curr_files = [fn]
                    curr_lang = lang
                else:
                    curr_files.append(fn)

            _outputs = ASR(curr_files,
                            batch_size=batch_size,
                            chunk_length_s=30,
                            generate_kwargs={'num_beams': 1, 'language': curr_lang},
                            return_timestamps=False)

            outputs.extend(_outputs)

            time_multilingual = time.time()-st
            print(f"[MultiLingualLongform Time]: {time_multilingual}")
        
        data['pred_text'] = [_['text'].strip() for _ in outputs]
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
    flash_attention = True if args.flash_attention == "yes" else False
    better_transformer = True if args.better_transformer == "yes" else False

    run(args.repo_path, flash_attention=flash_attention, better_transformer=better_transformer, batch_size=args.batch_size, eval_mp3=eval_mp3, eval_multilingual=eval_multilingual)