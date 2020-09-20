from  SinkhornTransformerLMGenerator import SinkhornTransformerLMGenerator
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    TransfoXLConfig
)
from tokenizers import ByteLevelBPETokenizer, Tokenizer, models, pre_tokenizers, decoders, processors
import torch
import os
import pickle
from BaseGru import BaseGru
from Seq2Seq import Seq2SeqRNN_attn


def get_config(args, vocabulary):
    if args.model_name in 'transfo-xl-wt103':
        config = TransfoXLConfig(vocab_size_or_config_json_file=args.vocab_size, cutoffs=[20000, 40000, 200000],
                                       d_model=512, d_embed=512, n_head=8, d_head=64, n_layer=12, d_inner=2048)
    else:
        config = AutoConfig.from_pretrained(args.model_name)

        config.vocab_size = args.vocab_size  # len(vocabulary.word2index.keys())
        config.n_positions = args.seq_size
      
    return config

def save_all(args, model, results):
    out_directory = args.out_directory

    if not os.path.exists(out_directory):
        os.mkdir(out_directory)
    with open(out_directory+'/args.pickle', 'wb') as handle:
        pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if any(mn in args.model_name for mn in ['xlnet', 'gpt2']):
        model.save_pretrained(out_directory)
    else:
        torch.save(model.state_dict(), out_directory+'/model')
    results.to_csv(out_directory+'/results.csv')
    print("Saved!")


def load_model(args, device):
    model = get_model(args, device)
    if 'sinkhorn' in args.model_name:
        model.load_state_dict(torch.load(args.out_directory+'/model'), strict=False)
    else:
        model = model.from_pretrained(args.out_directory)
    return model

def get_tokenizer(path):

    # Load a BPE Model
    vocab = path+"-vocab.json"
    merges = path+"-merges.txt"
    print('Vocab File: '+vocab)
    print('Merges file: '+merges)
    bpe = ByteLevelBPETokenizer(vocab_file=vocab, merges_file= merges)

    return bpe


def get_model(args, device, vocabulary=None, config=None):
    if any(mn in args.model_name for mn in args.transformers_models):
        if config is None:
            config = get_config(args, vocabulary)
            config.n_ctx = args.seq_size
            
        model = AutoModelWithLMHead.from_config(config)
        model = model.to(device)
    elif args.model_name in 'sinkhorn':
        model = SinkhornTransformerLMGenerator(
            num_tokens=args.vocab_size,
            dim=1024,
            heads=8,
            depth=12,
            bucket_size=20,  # size of the buckets
            causal=False,  # auto-regressive or not
            n_sortcut=2,  # use sortcut to reduce memory complexity to linear
            ff_chunks=10,  # feedforward chunking, from Reformer paper
            reversible=True,  # make network reversible, from Reformer paper
            ff_dropout=0.1,  # feedforward dropout
            attn_dropout=0.1,  # post attention dropout
            attn_layer_dropout=0.1,  # post attention layer dropout
            layer_dropout=0.1,  # add layer dropout, from 'Reducing Transformer Depth on Demand' paper
            weight_tie=True,  # tie layer parameters, from Albert paper
            emb_dim=128,  # embedding factorization, from Albert paper
            ff_glu=True,  # use GLU in feedforward, from paper 'GLU Variants Improve Transformer'
            n_local_attn_heads=2,
            # replace N heads with local attention, suggested to work well from Routing Transformer paper
            max_seq_len=args.seq_size
        ).to(device)
    elif args.model_name in 'gru':
        model = BaseGru(args.vocab_size, 1024, args.batch_size)
    elif args.model_name in 'seq2seq':
        model = Seq2SeqRNN_attn(256, 256, 1024, 100, 3, args.vocab_size+1, args.vocab_size+2, args.vocab_size)

    return model
