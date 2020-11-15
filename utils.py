import torch
import logging
import numpy as np

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def zero_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if h is None:
        return None
    if isinstance(h, torch.Tensor):
        return h * 0
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    # if args.cuda:
    #     data = data.cuda()
    #     target = target.cuda()
    return data, target

def read_txt_embeddings(emb_path,w2v = True , full_vocab = False,max_vocab = 200000, conc = False):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []
    conc_nouns = get_conc() if conc else None

    with open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):

            if i == 0 and w2v :
                split = line.split()
                assert len(split) == 2
                emb_dim = int(split[1])
                embs_size = int(split[0])
            else:
                word, vect = line.rstrip().split(' ', 1)
                if not full_vocab:
                    word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    if full_vocab:
                        logging.warning(f"Word {word} found twice in embedding file")
                elif conc:
                    if word in conc_nouns:
                        word2id[word] = len(word2id)
                        vectors.append(vect)
                else:
                    if w2v :
                        if not vect.shape == (emb_dim,):
                            logging.warning("Invalid dimension (%i) for %s word '%s' in line %i."
                                        % (vect.shape[0], 'source' if source else 'target', word, i))
                            continue
                        assert vect.shape == (emb_dim,), i
                    
                    word2id[word] = len(word2id)
                    vectors.append(vect)
            if max_vocab > 0 and len(word2id) >= max_vocab and not full_vocab:
                break
    assert len(word2id) == len(vectors)
    logging.info("Loaded %i pre-trained word embeddings." % len(vectors))

    embeddings = np.array(vectors)
    words = list(word2id.keys())
    #w2e = {w : e for w,e in zip(words,embeddings)}

    return words,embeddings,emb_dim,embs_size