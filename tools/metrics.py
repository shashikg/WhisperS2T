import jiwer
import editdistance
import diff_match_patch

import numpy as np
from tqdm import tqdm
from nltk import ngrams
from jiwer.transformations import wer_default


def WER(references, hypotheses):
    measures = jiwer.compute_measures(references, 
                                      hypotheses,
                                      truth_transform=wer_default, 
                                      hypothesis_transform=wer_default)
    
    out = {
        'WER': round(100*measures['wer'], 2),
        'IER': round(100*measures['insertions']/(measures['hits']+measures['substitutions']+measures['deletions']), 2),
        'DER': round(100*measures['deletions']/(measures['hits']+measures['substitutions']+measures['deletions']), 2),
        'SER': round(100*measures['substitutions']/(measures['hits']+measures['substitutions']+measures['deletions']), 2)
    }
    
    return out

def CER(references, hypotheses):
    errors = 0
    words = 0

    for h, r in zip(hypotheses, references):
        h_list = list(h)
        r_list = list(r)
        words += len(r_list)
        errors += editdistance.eval(h_list, r_list)

    return round(100*errors/words, 2)

def NGramDuplicates(text, ngram_size=5):
    all_ngrams = list(ngrams(text.split(), ngram_size))
    return len(all_ngrams)-len(set(all_ngrams))

def NGramInsertions(references, hypotheses, ngram_size=5):

    repeated_ngrams = 0
    for r, h in zip(references, hypotheses):
        all_ngrams = list(ngrams(r.split(), ngram_size))
        ref_counts = {}
        for ngram in all_ngrams:
            try:
                ref_counts[ngram] += 1
            except:
                ref_counts[ngram] = 1

        all_ngrams = list(ngrams(h.split(), ngram_size))
        hyp_counts = {}
        for ngram in all_ngrams:
            try:
                hyp_counts[ngram] += 1
            except:
                hyp_counts[ngram] = 1

        for k, v in hyp_counts.items():
            if (v > 1) and (ref_counts.get(k, 1) < v):
                repeated_ngrams += (v-ref_counts.get(k, 1))

    return repeated_ngrams

def evaluate(references, hypotheses, cer=False, ngram_size=5):
    scores = WER(references, hypotheses)
    if cer:
        scores.update({'CER': CER(references, hypotheses), f'{ngram_size}-GramInsertions': NGramInsertions(references, hypotheses, ngram_size=ngram_size)})
    else:
        scores.update({f'{ngram_size}-GramInsertions': NGramInsertions(references, hypotheses,  ngram_size=ngram_size)})
        
    return scores

def word_alignment_accuracy_single(references, hypotheses, collar=0.2):
    # Find diffs between ref and hyp
    r_list = [_['word'].replace(" ", "_") for _ in references]
    h_list = [_['word'].replace(" ", "_") for _ in hypotheses]
    
    orig_words = '\n'.join(r_list) + '\n'
    pred_words = '\n'.join(h_list) + '\n'
    
    diff = diff_match_patch.diff_match_patch()
    diff.Diff_Timeout = 0
    orig_enc, pred_enc, enc = diff.diff_linesToChars(orig_words, pred_words)
    diffs = diff.diff_main(orig_enc, pred_enc, False)
    diff.diff_charsToLines(diffs, enc)
    
    diffs_post = [(d[0], d[1].replace('\n', ' ').strip().split()) for d in diffs]

    # Find words which got HIT and their matching
    r_idx, h_idx = 0, 0
    word_idx_match = {}
    for case, words in diffs_post:
        if case == -1:
            r_idx += len(words)
        elif case == 1:
            h_idx += len(words)
        else:
            for _ in words:
                word_idx_match[r_idx] = h_idx
                r_idx += 1
                h_idx += 1


    # Find words whose alignments overlap with each other
    overlapped_words = 0
    within_collar_words = 0
    for r_idx, h_idx in word_idx_match.items():
        if (hypotheses[h_idx]['start']<references[r_idx]['end']) and (hypotheses[h_idx]['end']>references[r_idx]['start']):
            overlapped_words += 1

        if (hypotheses[h_idx]['start']>=references[r_idx]['start']-collar) and (hypotheses[h_idx]['end']<=references[r_idx]['end']+collar):
            within_collar_words += 1

    
    results = {
        'acc_overlapped': round(100*overlapped_words/len(word_idx_match), 2),
        'acc_within_collar': round(100*within_collar_words/len(word_idx_match), 2),
        'overlapped_words': overlapped_words,
        'within_collar_words': within_collar_words,
        'total_hit_words': len(word_idx_match),
    }

    return results

def word_alignment_accuracy(references, hypotheses, collar=0.2):
    overlapped_words = 0
    within_collar_words = 0
    total_hit_words = 0

    for r, h in tqdm(zip(references, hypotheses), total=len(references)):
        res = word_alignment_accuracy_single(r, h, collar=collar)
        overlapped_words += res['overlapped_words']
        within_collar_words += res['within_collar_words']
        total_hit_words += res['total_hit_words']

    results = {
        'acc_overlapped': round(100*overlapped_words/total_hit_words, 2),
        'acc_within_collar': round(100*within_collar_words/total_hit_words, 2),
        'overlapped_words': overlapped_words,
        'within_collar_words': within_collar_words,
        'total_hit_words': total_hit_words,
    }

    return results