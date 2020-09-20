import itertools
import random
import tokenize
from io import StringIO

import numpy as np
import torch.nn.functional as F

# from fastai.text import *


pad_token, pad_id = "§PAD§", 0
oov_token, oov_id = "§OOV§", 1
indent_token = "§<indent>§"
dedent_token = "§<dedent>§"
number_token = "§NUM§"


def preprocess(tokentype, tokenval):
    if tokentype == tokenize.NUMBER:
        return number_token

    elif tokentype == tokenize.INDENT:
        return indent_token

    elif tokentype == tokenize.DEDENT:
        return dedent_token

    return tokenval


def read_file(args, filename):
    data = []
    with open(filename, 'r') as f:
        content = f.read()
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    if not content.endswith('\n'):
        content += '\n'
    if args.use_python_vocabulary:
        tokens = tokenize.generate_tokens(StringIO(content).readline)
        data.append([preprocess(tokenType, tokenVal) for tokenType, tokenVal, _, _, _
                     in tokens
                     if tokenType != tokenize.COMMENT and
                     not tokenVal.startswith("'''") and
                     not tokenVal.startswith('"""') and
                     (tokenType == tokenize.DEDENT or tokenVal != "")])
        data = list(itertools.chain(*data))
    else:
        data = remove_comments_and_docstrings(content)
    return data


def closest_number(n, m):
    # Find the quotient
    q = int(n / m)

    # 1st possible closest number
    n1 = m * q

    # 2nd possible closest number
    if (n * m) > 0:
        n2 = (m * (q + 1))
    else:
        n2 = (m * (q - 1))

    # if true, then n1 is the required closest number
    if abs(n - n1) < abs(n - n2):
        return n1

    # else n2 is the required closest number
    return n2


def tokenize_text(text):
    tokenized = list(map(lambda x: preprocess(x[0], x[1]), list(tokenize.generate_tokens(StringIO(text).readline))))
    end_index = tokenized.index('')
    return tokenized[:end_index]


def get_data_from_files(files, batch_size, seq_size, vocab, n_out=1, limit=None, shuffle=False):
    X = []
    y = []
    if shuffle:
        random.shuffle(files)
    if limit is not None:
        files = files[:limit]

    for datafile in files:
        batch_num = 0
        text = read_file(datafile)
        int_text = [vocab.to_index(w) for w in text]
        # int_text = text
        num_batches = int(len(int_text) / (seq_size * batch_size))
        for i in range(len(int_text)):
            if (i + seq_size + n_out) < len(int_text[i + batch_size * batch_num:]):

                X.append(int_text[i: i + seq_size])  # +  batch_size * batch_num
                y.append(int_text[i + seq_size: i + seq_size + n_out])
                if len(X) > batch_size:
                    X_out = np.array(X)
                    y_out = np.array(y)
                    X = []
                    y = []
                    batch_num += 1
                    yield X_out, y_out


def generate_easy_sequence(inp, seq_size, batch_size):
    num_batches = (len(inp) - 1 - seq_size) // batch_size + 1
    batches = []
    for j in range(num_batches - 1):
        X = []
        y = []
        for i in range(batch_size):
            if ((i + seq_size + batch_size * j) < len(inp)):
                X.append(inp[i + batch_size * j: i + seq_size + batch_size * j])
                y.append([inp[i + seq_size + batch_size * j]])

        batches.append([X, y])
    return batches


def remove_comments_and_docstrings(source):
    """
    Returns 'source' minus comments and docstrings.
    """
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        # The following two conditionals preserve indentation.
        # This is necessary because we're not using tokenize.untokenize()
        # (because it spits out code with copious amounts of oddly-placed
        # whitespace).
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:
                    # Note regarding NEWLINE vs NL: The tokenize module
                    # differentiates between newlines that start a new statement
                    # and newlines inside of operators such as parens, brackes,
                    # and curly braces.  Newlines inside of operators are
                    # NEWLINE and newlines that start new code are NL.
                    # Catch whole-module docstrings:
                    if start_col > 0:
                        # Unlabelled indentation means we're inside an operator
                        out += token_string
                    # Note regarding the INDENT token: The tokenize module does
                    # not label indentation inside of an operator (parens,
                    # brackets, and curly braces) as actual indentation.
                    # For example:
                    # def foo():
                    #     "The spaces before this docstring are tokenize.INDENT"
                    #     test = [
                    #         "The spaces before this string do not get a token"
                    #     ]

        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join([line for line in out.splitlines() if line.strip()])
    return out
