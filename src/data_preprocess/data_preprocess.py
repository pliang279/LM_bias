from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
import torch
import random
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_source", type=str, default="glove",
                        help="choose the source of word embedding, e.g. glove, gpt2")
    parser.add_argument("--by_pca", type=bool, default=True, help="whether to use PCA to obatin the bias subspace")
    parser.add_argument("--num_components", type=int, default=5, help="number of components of the bias subspace")
    parser.add_argument("--save_subspace", type=bool, default=False, help="whether to save the bias subspace")
    parser.add_argument("--save_path", type=str, default="../../data/bias_subspace/", help="whether to save the bias subspace")
    args = parser.parse_args()
    return args


# obtain glove biased tokens
def load_word_vectors(fname):
    model = KeyedVectors.load_word2vec_format(fname, binary=False)
    vecs = model.vectors
    words = list(model.vocab.keys())
    return model, vecs, words


def doPCA(pairs, num_components=10):
    matrix = []
    for a, b in pairs:
        center = (a + b) / 2
        norm_a = a - center
        norm_b = b - center
        matrix.append(norm_a)
        matrix.append(norm_b)
    matrix = np.array(matrix)
    pca = PCA(n_components=num_components, svd_solver="full")
    pca.fit(matrix)
    return pca


def bias_subspace(embed, source=None, save_subspace=False, save_path="", by_pca=True, num_components=5):
    # define gender direction
    if by_pca:
        if source == "glove":
            pairs = [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"],
                     ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"], ["mary", "john"]]
        elif source == "gpt2":
            pairs = [["Ġwoman", "Ġman"], ["Ġgirl", "Ġboy"], ["Ġshe", "Ġhe"], ["Ġmother", "Ġfather"],
                     ["Ġdaughter", "Ġson"], ["Ġfemale", "Ġmale"], ["Ġher", "Ġhis"], ["Ġherself", "Ġhimself"],
                     ["ĠMary", "ĠJohn"], ["Ġmom", "Ġdad"], ["Ġgal", "Ġguy"]]
        pair = []
        for p in pairs:
            pair.append((embed[p[0]], embed[p[1]]))
        pca = doPCA(pair)
        bias_direction = pca.components_[0]  # man - woman
        bias_subspace = pca.components_[:num_components]
        print("pca explained variance ratio: ", pca.explained_variance_ratio_)
        if save_subspace:
            np.save(source + "_bias_subspace", bias_direction)
            np.save(source + "_bias_direction", bias_subspace)
    else:
        bias_direction = embed["he"] - embed["she"]
        if save_subspace:
            np.save(source + "_bias_subspace", bias_direction)

    return bias_direction


def data_preprocess():
    args = get_args()

    if args.embed_source not in ["glove", "gpt2"]:
        print("Embedding source should be either glove or gpt2. Or you can download other embeddings by yourself.")
        return

    if args.embed_source == "glove":
        embed, vecs, words = load_word_vectors(fname="../../data/embeddings/vecs.filtered.txt")
    else:
        embed, vecs, words = load_word_vectors(fname="../../data/embeddings/gpt2_embedding.txt")

    bias_direction = bias_subspace(embed=embed, source=args.embed_source, save_subspace=args.save_subspace,
                                   save_path=args.save_path, by_pca=args.by_pca, num_components=args.num_components)
    if args.embed_source == "glove":
        if bias_direction.dot(embed["woman"] - embed["man"]) > 0:
            bias_direction = -bias_direction
    else:
        if bias_direction.dot(embed["Ġwoman"] - embed["Ġman"]) > 0:
            bias_direction = -bias_direction

    # projection on the gender direction, and got top n biased token
    n = 2500
    group1 = embed.similar_by_vector(bias_direction, topn=n, restrict_vocab=None)    # male biased
    group2 = embed.similar_by_vector(-bias_direction, topn=n, restrict_vocab=None)    # female biased
    male_tokens, male_scores = list(zip(*group1))
    female_tokens, female_scores = list(zip(*group2))
    print("top 100 male tokens:", male_tokens[:100])
    print("top 100 male projection values:", male_scores[:100])
    print()
    print("top 100 female tokens:", female_tokens[:100])
    print("top 100 female projection values:", female_scores[:100])
    print()

    # this threshold should be tuned according to the used embedding
    threshold_male, threshold_female = 0.24, 0.25
    female_biased_token = np.array(female_tokens)[np.array(list(female_scores)) > threshold_female]
    male_biased_token = np.array(male_tokens)[np.array(list(male_scores)) > threshold_male]

    male_biased_token_set = set(male_biased_token)
    female_biased_token_set = set(female_biased_token)
    if args.embed_source == "glove":
        pairs = [['woman', 'man'], ['girl', 'boy'], ['she', 'he'], ['mother', 'father'], ['daughter', 'son'],
                 ['female', 'male'], ['her', 'his'], ['herself', 'himself'], ['mary', 'john'], ['mom', 'dad'],
                 ['gal', 'guy']]
    else:
        pairs = [["Ġwoman", "Ġman"], ["Ġgirl", "Ġboy"], ["Ġshe", "Ġhe"], ["Ġmother", "Ġfather"],
                 ["Ġdaughter", "Ġson"], ["Ġfemale", "Ġmale"], ["Ġher", "Ġhis"], ["Ġherself", "Ġhimself"],
                 ["ĠMary", "ĠJohn"], ["Ġmom", "Ġdad"], ["Ġgal", "Ġguy"]]
    for p in pairs:
        female_biased_token_set.add(p[0])
        male_biased_token_set.add(p[1])
        if args.embed_source == "glove":
            female_biased_token_set.add(p[0].capitalize())
            male_biased_token_set.add(p[1].capitalize())
    print("male biased tokens in " + args.embed_source, male_biased_token.shape,
          "female biased tokens in " + args.embed_source, female_biased_token.shape)
    print(male_biased_token_set)
    print(female_biased_token_set)

    # there are some neutral tokens in the biased token set, we can remove them manually or estimate better bias direction
    if args.embed_source == "glove":
        male_filter_list = ["general", "drafted"]
        female_filter_list = ["sassy"]
    else:
        male_filter_list, female_filter_list = [], []
    for token in male_filter_list:
        male_biased_token_set.remove(token)
    for token in female_filter_list:
        female_biased_token_set.remove(token)

    if args.embed_source == "gpt2":
        # when using gpt2 embedding, we can also use tokenizer to find bias specific sentences
        male_biased_token_set = set([x.replace("Ġ", "") for x in male_biased_token_set])
        female_biased_token_set = set([x.replace("Ġ", "") for x in female_biased_token_set])

    corpus = ["../../data/text_corpus/reddit.txt", "../../data/text_corpus/meld.txt", "../../data/text_corpus/news_100.txt",
              "../../data/text_corpus/news_200.txt", "../../data/text_corpus/sst.txt", "../../data/text_corpus/wikitext.txt",
              "../../data/text_corpus/yelp_review_1mb.txt", "../../data/text_corpus/yelp_review_5mb.txt",
              "../../data/text_corpus/yelp_review_10mb.txt"]
    male_biased_sent, female_biased_sent = [], []  # save gender biased sentences
    male_biased_sent_clip, female_biased_sent_clip = [], []  # save gender biased sentences clipped version
    neut_sent_clip = []
    male_token_ident, female_token_ident = set(), set()  # save tokens which are used to identify the gender of the sentences
    for cor in corpus:
        with open(cor, 'r') as f:
            text_corpus = f.read()
        text_corpus = text_corpus.split('\n')
        count = 0
        tmp_neut = []

        for sent in text_corpus:
            male_flag, female_flag = False, False  # indicate whether biased token appears in this sentence
            idx = -1
            tokens = sent.split(' ')
            if len(tokens) < 5:  # if the sentence is too short, skip
                continue
            for token in tokens:
                if token in male_biased_token_set:  # find male definitional token
                    male_flag = True
                    male_token_ident.add(token)
                    idx = tokens.index(token)
                if token in female_biased_token_set:  # find female definitional token
                    female_flag = True
                    female_token_ident.add(token)
                    idx = tokens.index(token)
                    if count < 20:
                        count += 1
                if male_flag and female_flag:  # both male and female appears
                    break

            if (not male_flag) and (not female_flag):  # neither male or female doesn't appear, view as neutral
                index = random.randint(4, len(tokens))  # start from 4th token
                tmp_neut.append(" ".join(tokens[:index]))
                continue

            if male_flag and female_flag:  # both male and female appears
                continue

            if male_flag:
                if sent not in male_biased_sent:  # remove duplicate sentence
                    male_biased_sent.append(sent)
                    index = random.randint(idx, len(tokens))
                    male_biased_sent_clip.append(" ".join(tokens[:index + 1]))

            if female_flag:
                if sent not in female_biased_sent:
                    female_biased_sent.append(sent)
                    index = random.randint(idx, len(tokens))
                    female_biased_sent_clip.append(" ".join(tokens[:index + 1]))

        tmp_neut = np.array(tmp_neut)
        tmp_neut = np.random.choice(tmp_neut, 1000, replace=False)
        neut_sent_clip += tmp_neut.tolist()

    print(len(male_biased_sent))
    print(len(female_biased_sent))
    print(len(neut_sent_clip))
    print(male_token_ident)
    print(female_token_ident)

    # np.savetxt("../../data/male_sentences.txt", male_biased_sent, fmt="%s")
    # np.savetxt("../../data/female_sentences.txt", female_biased_sent, fmt="%s")
    # np.savetxt("../../data/neut_sentences.txt", neut_sent_clip, fmt="%s")
    # np.savetxt("../../data/male_sentences_clip.txt", male_biased_sent_clip, fmt="%s")
    # np.savetxt("../../data/female_sentences_clip.txt", female_biased_sent_clip, fmt="%s")


if __name__ == '__main__':
    data_preprocess()
