import os
from typing import Collection, Generator
import spacy
from spacy.attrs import ORTH
from collections import Counter
from multiprocessing import Process, Queue
import numpy as np
import re
import html
import gc
import pathlib