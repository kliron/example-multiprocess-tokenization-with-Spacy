#!/usr/bin env python3

import os
from typing import Collection, List, Generator, Callable, Match
import spacy
from spacy.attrs import ORTH
from collections import Counter
from multiprocessing import Process, Queue, cpu_count
import numpy as np
import re
import html
from pathlib import Path
import tqdm

root_dir = Path(Path.home(), 'Data/corpora')
txt_dir = Path(root_dir, 'texts')
tokens_dir = Path(root_dir, 'tokens')
tokens_out_file_format = os.path.join(tokens_dir, 'tokens{}.txt')
words_file = Path(tokens_dir, 'words.txt')  # Flat text file containing _all_ tokens in descending order of frequency
tokens_file = Path(tokens_dir, 'tokens.txt')

# Special tokens
BOS, EOS, FLD, UNK, PAD, TK_MAJ, TK_UP, TK_REP, TK_WREP = 'xxbos', 'xxeos', 'xxfld', 'xxunk', 'xxpad', 'xxmaj', \
                                                          'xxup', 'xxrep', 'xxwrep'

text_spec_tok = [UNK, PAD, BOS, EOS, FLD, TK_MAJ, TK_UP, TK_REP, TK_WREP]


def spec_add_spaces(t: str) -> str:
    """Add spaces around / and # in `t`. \n"""
    return re.sub(r'([/#\n])', r' \1 ', t)


def rm_useless_spaces(t: str) -> str:
    """Remove multiple spaces in `t`."""
    return re.sub(' {2,}', ' ', t)


def replace_rep(t: str) -> str:
    """Replace repetitions at the character level in `t`."""
    def _replace_rep(m: Match) -> str:
        c, cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '
    re_rep = re.compile(r'(\S)(\1{3,})')
    return re_rep.sub(_replace_rep, t)


def replace_wrep(t: str) -> str:
    """Replace word repetitions in `t`."""
    def _replace_wrep(m: Match) -> str:
        c, cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '
    re_wrep = re.compile(r'(\b\w+\W+)(\1{3,})')
    return re_wrep.sub(_replace_wrep, t)


def fix_html(x: str) -> str:
    """List of replacements from html strings in `x`."""
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', UNK).replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace(' @,@ ', ',').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


def replace_all_caps(x: Collection[str]) -> Collection[str]:
    """Replace tokens in ALL CAPS in `x` by their lower version and add `TK_UP` before."""
    res = []
    for t in x:
        if t.isupper() and len(t) > 1:
            res.append(TK_UP)
            res.append(t.lower())
        else:
            res.append(t)
    return res


def deal_caps(x: Collection[str]) -> Collection[str]:
    """Replace all Capitalized tokens in `x` by their lower version and add `TK_MAJ` before."""
    res = []
    for t in x:
        if t == '':
            continue
        if t[0].isupper() and len(t) > 1 and t[1:].islower():
            res.append(TK_MAJ)
        res.append(t.lower())
    return res


# Preprocess and postprocess rules for tokenization
pre_rules = [fix_html, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces]
post_rules = [replace_all_caps, deal_caps]


def batch_filenames(batch_size: int) -> List[List[str]]:
    """Recursively reads all files under `txt_dir` and returns batch_size 
    number of lists of pathnames for processing."""
    file_paths = []
    for root, dirs, files in os.walk(txt_dir):
        for f in files:
            file_paths.append(os.path.join(root, f))
    np.random.shuffle(file_paths)
    return np.array_split(file_paths, batch_size)


def text_gen(path_list: Collection[str]) -> Generator[str, None, None]:
    """Reads files from path_list and returns their text content sequentially
    after applying pre-processing rules."""
    for path in path_list:
        with open(path, 'r') as r:
            txt = r.read()
            for fn in pre_rules:
                txt = fn(txt)
            yield txt                


def process_files(lang: str,
                  path_list: Collection[str],                  
                  output_file: str,
                  data_queue: Queue,
                  progress_queue: Queue,
                  batch_size: int = 10000):
    """Extracts tokens from each file in path_list and keeps a Counter of all tokens.
    Writes tokens as space separated strings in `output_file`. Reports progress on `progress_queue`
    When done, puts its Counter on the shared queue `queue`.
    `batch_size` is the number of text batches to read at once. Depending on how large text chunks
    your files contain and how much free memory you have, you might want to adjust the default"""
    nlp = spacy.blank(lang, disable=["parser", "tagger", "ner"])
    # This is where we would add any custom parse rules (via eg. `prefix_search`, etc) to the tokenizer.
    # See https://spacy.io/api/tokenizer#init
    for w in text_spec_tok:
        nlp.tokenizer.add_special_case(w, [{ORTH: w}])

    texts = text_gen(path_list)
    counts = Counter()

    with open(output_file, 'w') as w:
        for docs in nlp.pipe(texts, batch_size=batch_size):
            tokens = [t.text for t in docs]
            for fn in post_rules:
                tokens = fn(tokens)
            w.write(' '.join(tokens))
            counts += Counter(tokens)
            progress_queue.put(1)

    data_queue.put(counts)


def postprocess(file: str):
    pass


def tokenize(lang: str, n_workers: int, tfile: str, postprocess_fn: Callable = postprocess):
    """
    1. Creates a shared queue and forks `n_workers` tokenizer processes. Waits for all processes to
    finish and then merges all worker counters. 
    2. Concatenates all token files into a single file `tfile` and removes the component files.
    3. Calls `postprocess` fn on the resulting file path.
    The resulting file should be a plaintext file where each line consists of space-separated tokens from a single processed 
    initial text file.
    """    
    batches = batch_filenames(batch_size=n_workers)
    output_files = []
    processes = []
    data_queue = Queue(maxsize=n_workers)
    progress_queue = Queue()
    counter = Counter()

    for i, batch in enumerate(batches):
        output_file = tokens_out_file_format.format(i+1)
        output_files.append(output_file)
        processes.append(Process(target=process_files,
                                 args=(lang, batch, output_file, data_queue, progress_queue)))
    for p in processes:
        p.start()

    # Report progress
    total = 0
    for b in batches:
        total += len(b)

    pbar = tqdm.tqdm(total=total)

    i = 0
    while i < total:
        _ = progress_queue.get()
        pbar.update()
        i += 1

    for _ in processes:
        c = data_queue.get()
        counter += c

    for p in processes:
        p.join()

    with open(words_file, 'w') as w:
        w.write(' '.join([items[0] for items in counter.most_common()]))

    postprocess_fn(tfile)


def main():
    print(f'Tokens file will be saved at {tokens_file}')
    tokenize(lang='sv',
             n_workers=cpu_count()-1,
             tfile=tokens_file)


if __name__ == '__main__':
    main()
