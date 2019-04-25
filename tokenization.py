#!/usr/bin env python3

import os
from typing import Collection, Generator
import spacy
from spacy.attrs import ORTH
from collections import Counter
from multiprocessing import Process, Queue, cpu_count
import numpy as np
import re
import html
import gc
import pathlib

txt_dir = os.path.join(pathlib.Path.home(), 'Data/corpora')
subdirs = [
    'fass',
    'lakartidningen',
    'radiology',
    'literaturbanken',
    'svwiki',
    'riksdagen',
    'riksdagen1',
    'riksdagen2',
    'riksdagen3',
    'riksdagen4',
    'riksdagen5',
    'riksdagen6',
    'riksdagen7',
    'riksdagen8',
    'riksdagen9'
]
tokens_out_file_format = os.path.join(txt_dir, 'tokens{}.txt')
# just a list of words sorted from most to least common
vocab_out_file = os.path.join(txt_dir, 'vocab.txt')

# Special tokens
BOS, EOS, FLD, UNK, PAD = 'xxbos', 'xxeos', 'xxfld', 'xxunk', 'xxpad'
TK_MAJ, TK_UP, TK_REP, TK_WREP = 'xxmaj', 'xxup', 'xxrep', 'xxwrep'

text_spec_tok = [UNK, PAD, BOS, EOS, FLD, TK_MAJ, TK_UP, TK_REP, TK_WREP]


def batch_filenames(batch_size: int) -> Collection[Collection[str]]:
    """Returns batch_size number of lists of pathnames to textfiles for processing."""
    file_paths = []
    for d in subdirs:
        dir_path = os.path.join(txt_dir, d)
        file_paths += [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.txt')]
    np.random.shuffle(file_paths)
    return np.array_split(file_paths, batch_size)


def text_gen(path_list: Collection[str]) -> Generator[str, None, None]:
    """Reads files from path_list and returns their text content sequentially."""
    for path in path_list:
        with open(path, 'r') as r:
            yield r.read()


def process_files(lang: str,
                  path_list: Collection[str],
                  queue: Queue,
                  output_file: str,
                  batch_size: int = 5000):
    """Extracts tokens from each file in path_list and keeps a Counter of all tokens.
    Writes tokens as space separated strings in `output_file`.
    When done, puts its Counter on the shared queue `queue`.
    `batch_size` is the number of text batches to read at once. Depending on how large text chunks
    your files contain and how much free memory you have, you might want to adjust the default"""
    pid = os.getpid()
    nlp = spacy.blank(lang, disable=["parser", "tagger", "ner"])
    # This is where we would add any exceptions, parse instructions (via eg. `prefix_search`, etc)
    # to the tokenizer. See https://spacy.io/api/tokenizer#init
    for w in text_spec_tok:
        nlp.tokenizer.add_special_case(w, [{ORTH: w}])

    texts = text_gen(path_list)
    counts = Counter()

    with open(output_file, 'w') as w:
        for docs in nlp.pipe(texts, batch_size=batch_size):
            gc.collect()
            tokens = [t.text for t in docs]
            counts += Counter(tokens)
            w.write(' '.join(tokens))

    queue.put(counts)
    print('Process {} finished.'.format(pid))


def main(lang: str, n_workers: int, max_vocab: int = 60000):
    """
    Creates a shared queue and forks `n_workers` tokenizer processes. Waits for all processes to
    finish and then merges all worker counters.
    WARNING! Don't put very large objects on multiprocessing.Queue. Queue.get is extremely slow. The
    performance of reading through IPC in python is atrocious.
    """
    queue = Queue(maxsize=n_workers)
    c = Counter()
    batches = batch_filenames(batch_size=n_workers)
    processes = []

    for i, batch in enumerate(batches):
        batch = batch
        processes.append(Process(target=process_files,
                                 args=(lang, batch, queue, tokens_out_file_format.format(i+1))))
    for p in processes:
        p.start()

    for _ in processes:
        counter = queue.get()
        c += counter

    for p in processes:
        p.join()

    with open(vocab_out_file, 'w') as w:
        print('Tokenizing done. Writing vocabulary.')
        w.write(' '.join([items[0] for items in c.most_common(max_vocab)]))

    print('All done.')


if __name__ == '__main__':
    main(lang='sv', n_workers=cpu_count()-1)
