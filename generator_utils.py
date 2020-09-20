import torch

from utils import tokenize_text, closest_number


def predict_beam(args, device, net, text, vocabulary, num_return_sequences=5, n_out=1):
    net.eval()

    normalized_text = text  # normalize_text(text)
    if args.use_python_vocabulary:
        tokens = tokenize_text(normalized_text)
        ix = torch.tensor([[vocabulary.to_index(w) for w in tokens]]).to(device)
    else:
        ix = torch.tensor([vocabulary.encode(normalized_text).ids[-100:]]).to(device)

    if any(mn in args.model_name for mn in args.transformers_models):
        output = net.generate(
            input_ids=ix,
            max_length=len(ix[0]) + n_out,
            num_beams=5,
            early_stopping=True,
            eos_token_id=None,
            num_return_sequences=num_return_sequences
            #pad_token_id=vocabulary.pad_id
        )
    else:
        cn = len(ix[0])//args.sinkhorn_bucket_size
        cn = cn*args.sinkhorn_bucket_size
        cn = min(cn, 100)
        ix = ix[:, -cn:]
        output = net.generate(
            input_ids=ix,
            max_length=len(ix[0]) + n_out,
            num_beams=5,
            early_stopping=True,
            eos_token_id=None,
            num_return_sequences=num_return_sequences,
            vocab_size=args.vocab_size
        )

    result = []
    for choice in output:
        if args.use_python_vocabulary:
            words = [vocabulary.to_word(x) for x in choice.tolist()]
            print(' '.join(words))
        else:
            words = vocabulary.decode(choice.tolist())
            print(words)
        #result.append(words[-n_out:])
       
        print('-' * 30)
    #return result


def predict_top_p(args, device, net, text, vocabulary, num_return_sequences=5, n_out=1):
    net.eval()

    normalized_text = text  # normalize_text(text)

    if args.use_python_vocabulary:
        tokens = tokenize_text(normalized_text)
        ix = torch.tensor([[vocabulary.to_index(w) for w in tokens]]).to(device)
    else:
        ix = torch.tensor([vocabulary.encode(normalized_text).ids[-100:]]).to(device)

    if any(mn in args.model_name for mn in args.transformers_models):
        output = net.generate(
            input_ids=ix,
            max_length=len(ix[0]) + n_out,
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            early_stopping=True
        )
    else:
        cn = len(ix[0])//args.sinkhorn_bucket_size
        cn = cn*args.sinkhorn_bucket_size
        cn = min(cn, 100)
        ix = ix[:, -cn:]
        output = net.generate(
            input_ids=ix,
            max_length=len(ix[0]) + n_out,
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            vocab_size=args.vocab_size
        )

    # result = []
    for choice in output:
        if args.use_python_vocabulary:
            words = [vocabulary.to_word(x) for x in choice.tolist()]
            print(' '.join(words))
        else:
            words = vocabulary.decode(choice.tolist())
            print(words)
        print('-' * 30)
    # return result


def predict_top_k(args, device, net, text, vocabulary, num_return_sequences=5, n_out=1):
    net.eval()

    normalized_text = text  # normalize_text(text)
    if args.use_python_vocabulary:
        tokens = tokenize_text(normalized_text)
        ix = torch.tensor([[vocabulary.to_index(w) for w in tokens]]).to(device)
    else:
        ix = torch.tensor([vocabulary.encode(normalized_text).ids[-100:]]).to(device)

    if any(mn in args.model_name for mn in args.transformers_models):
        output = net.generate(
            input_ids=ix,
            max_length=len(ix[0]) + n_out,
            top_k=50,
            do_sample=True,
            num_return_sequences=num_return_sequences
            #pad_token_id=vocabulary.pad_id
        )

    else:
        cn = len(ix[0])//args.sinkhorn_bucket_size
        cn = cn*args.sinkhorn_bucket_size
        cn = min(cn, 100)
        ix = ix[:, -cn:]
        print(cn)
        output = net.generate(
            input_ids=ix,
            max_length=len(ix[0]) + n_out,
            top_k=50,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            vocab_size=args.vocab_size
        )
    #result = []
    for choice in output:
        if args.use_python_vocabulary:
            words = [vocabulary.to_word(x) for x in choice.tolist()]
            print(' '.join(words))
        else:
            words = vocabulary.decode(choice.tolist())
            print(words)
        print('-' * 30)
    #return result
