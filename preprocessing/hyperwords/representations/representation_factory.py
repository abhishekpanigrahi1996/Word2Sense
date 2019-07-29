from embedding import SVDEmbedding, EnsembleEmbedding, Embedding, discriminative_embedding, discriminative_SGNS, Projective_embedding
from explicit import PositiveExplicit


def create_representation(args):
    rep_type = args['<representation>']
    path = args['<representation_path>']
    neg = int(args['--neg'])
    w_c = args['--w+c']
    eig = float(args['--eig'])
    
    if rep_type == 'PPMI':
        if w_c:
            raise Exception('w+c is not implemented for PPMI.')
        else:
            return PositiveExplicit(path, True, neg)
        
    elif rep_type == 'SVD':
        if w_c:
            return EnsembleEmbedding(SVDEmbedding(path, False, eig, False), SVDEmbedding(path, False, eig, True), True)
        else:
            return SVDEmbedding(path, True, eig)
        
    elif rep_type == 'SGNS':
        if w_c:
            return EnsembleEmbedding(Embedding(path + '.words', False), Embedding(path + '.contexts', False), True)
        else:
            return Embedding(path + '.words', True)
    elif rep_type == 'discriminative':
        return discriminative_embedding(path, True, eig)
    elif rep_type == 'discriminative_SGNS':
        return  discriminative_SGNS(path, True)

    elif rep_type == 'projective':
        return Projective_embedding(path)
