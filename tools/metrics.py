import jiwer
import editdistance
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