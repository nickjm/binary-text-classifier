from tqdm import tqdm
import argparse

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def main(args):
    corpus_len = file_len(args.corpus)
    embeddings_len = file_len(args.embeddings)
    with open(args.embeddings, 'r') as embeddings:
        with open(args.corpus, 'r') as corpus:
            with open(args.save_path, 'a') as pruned_embeddings:
                vocabulary = set()
                print("Loading corpus...")
                with tqdm(total=corpus_len) as pbar:
                    for line in corpus:
                        words = line.split('\t')[1].split()
                        for word in words:
                            vocabulary.add(word)
                        pbar.update()
                print("Pruning embeddings...")
                with tqdm(total=embeddings_len) as pbar:
                    for line in embeddings:
                        word = line.split()[0]
                        if word in vocabulary:
                            pruned_embeddings.write(line)
                        pbar.update()

def get_args():
    parser = argparse.ArgumentParser(description='Prune line separated embedding file to only those tokens found in a given text corpus in the format (label, sentence pairs)')
    parser.add_argument('--embeddings', type=str, help='path to embeddings')
    parser.add_argument('--corpus', type=str, help='path to corpus')
    parser.add_argument('--save_path', type=str, help='path to save pruned embeddings')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
