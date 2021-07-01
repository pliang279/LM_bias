from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import numpy as np
import torch
import random

# obtain glove biased tokens
def load_word_vectors(fname):
    model = KeyedVectors.load_word2vec_format(fname, binary=False)
    vecs = model.vectors
    words = list(model.vocab.keys())
    return model, vecs, words


def doPCA(pairs, num_components=10):
    matrix = []
    for a, b in pairs:
        center = (a + b)/2
        norm_a = a - center
        norm_b = b - center
        matrix.append(norm_a)
        matrix.append(norm_b)
    matrix = np.array(matrix)
    pca = PCA(n_components=num_components, svd_solver="full")
    pca.fit(matrix) # Produce different results each time...
    return pca


def bias_subspace(model, source=None, save_subspace=False, by_pca = True):
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
            pair.append((model[p[0]], model[p[1]]))
        pca = doPCA(pair)
        bias_direction = pca.components_[0]  # man - woman
        bias_subspace = pca.components_[:5]
        print(pca.explained_variance_ratio_)
        if save_subspace:
            np.save(source + "_bias_subspace", bias_direction)
            np.save(source + "_bias_direction", bias_subspace)
    else:
        bias_direction = model["he"] - model["she"]
        if save_subspace:
            np.save(source + "_bias_subspace", bias_direction)

    return bias_direction


def data_preprocess(embed_source="glove"):
    if embed_source == "glove":
        model, vecs, words = load_word_vectors(fname="../../nullspace_projection/data/embeddings/vecs.filtered.txt")
    else:
        model, vecs, words = load_word_vectors(fname="../../nullspace_projection/data/embeddings/gpt2_embedding.txt")
    bias_direction = bias_subspace(model=model, source=embed_source, save_subspace=False, by_pca=True)

    # projection on the gender direction, and got top n biased token
    n = 2500
    group1 = model.similar_by_vector(bias_direction, topn=n, restrict_vocab=None)    # male biased
    group2 = model.similar_by_vector(-bias_direction, topn=n, restrict_vocab=None)    # female biased
    male_tokens, male_scores = list(zip(*group1))
    female_tokens, female_scores = list(zip(*group2))
    print(male_tokens[:100])
    print(male_scores[:100])
    # print("top 10 scores for male biased token: ", male_scores[:10])
    # print(male_tokens[2400:2500])
    # print("lowest 10 scores for male biased token: ", male_scores[2490:2500])
    print()
    print(female_tokens[:100])
    print(female_scores[:100])
    # print("top 10 scores for female biased token: ", female_scores[:10])
    # print("lowest 10 scores for female biased token: ", female_scores[2490:2500])
    female_biased_token = np.array(female_tokens)[np.array(list(female_scores)) > 0.25]
    male_biased_token = np.array(male_tokens)[np.array(list(male_scores)) > 0.24]
    print()
    print("male biased tokens in " + embed_source, male_biased_token.shape, "female biased tokens in " + embed_source, female_biased_token.shape)

    male_biased_token_set = set(male_biased_token)
    female_biased_token_set = set(female_biased_token)
    pairs = [['woman', 'man'], ['girl', 'boy'], ['she', 'he'], ['mother', 'father'], ['daughter', 'son'],
             ['female', 'male'], ['her', 'his'], ['herself', 'himself'], ['mary', 'john'], ['mom', 'dad'],
             ['gal', 'guy'], ['her', 'him'], ['Woman', 'Man'], ['Girl', 'Boy'], ['She', 'He'], ['Mother', 'Father'],
             ['Daughter', 'Son'], ['Female', 'Male'], ['Her', 'His'], ['Mary', 'John'], ['Mom', 'Dad'], ['Gal', 'Guy']]
    # for p in [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"], ["mary", "john"]]:
    for p in pairs:
        female_biased_token_set.add(p[0])
        male_biased_token_set.add(p[1])
    female_biased_token_set.add("mom")
    male_biased_token_set.add("dad")
    print(len(male_biased_token_set), male_biased_token_set)
    print()
    print(len(female_biased_token_set), female_biased_token_set)
    male_biased_token_set.remove("general")
    male_biased_token_set.remove("drafted")
    female_biased_token_set.remove("sassy")

    corpus = ["data/text_corpus/reddit.txt", "data/text_corpus/meld.txt", "data/text_corpus/news_100.txt", "data/text_corpus/news_200.txt",
              "data/text_corpus/sst.txt", "data/text_corpus/wikitext.txt", "data/text_corpus/yelp_review_1mb.txt",
              "data/text_corpus/yelp_review_5mb.txt", "data/text_corpus/yelp_review_10mb.txt"]
    male_biased_sent, female_biased_sent = [], []  # save gender biased sentences
    male_biased_sent_clip, female_biased_sent_clip = [], []  # save gender biased sentences clipped version
    neut_sent_clip = []
    male_token_ident, female_token_ident = set(), set()  # save tokens which are used to identify the gender of the sentences
    for cor in corpus:
        with open(cor, 'r') as f:
            text_corpus = f.read()
        text_corpus = text_corpus.split('\n')
        # print()
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
                        # print(token, ": ", sent)
                        count += 1
                if male_flag and female_flag:  # both male and female appears
                    # print("both male and female appears", sent)
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

    # np.savetxt("data/male_sentences.txt", male_biased_sent, fmt="%s")
    # np.savetxt("data/female_sentences.txt", female_biased_sent, fmt="%s")
    # np.savetxt("data/neut_sentences.txt", neut_sent_clip, fmt="%s")
    # np.savetxt("data/male_sentences_clip.txt", male_biased_sent_clip, fmt="%s")
    # np.savetxt("data/female_sentences_clip.txt", female_biased_sent_clip, fmt="%s")


if __name__ == '__main__':
    data_preprocess(embed_source="glove")   # gpt2
