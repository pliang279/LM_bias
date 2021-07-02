# Some codes are from https://github.com/shauli-ravfogel/nullspace_projection

import transformers
from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}
model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
tokenizer = tokenizer_class.from_pretrained("gpt2")
model = model_class.from_pretrained("gpt2")

import numpy as np
import torch
import time

import sys
sys.path.append("../../nullspace_projection/src")
import classifier
import debias
import gensim
import codecs
import json
from gensim.models.keyedvectors import Word2VecKeyedVectors
from gensim.models import KeyedVectors
import random
import sklearn
from sklearn import model_selection
from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import scipy
from scipy import linalg
from scipy.stats.stats import pearsonr
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from torch.nn import functional as F


def load_data():
    # load clipped sentences
    sample = 5000
    male_sent = np.loadtxt("../../data/male_sentences_clip.txt", dtype=str, delimiter="\n")
    female_sent = np.loadtxt("../../data/female_sentences_clip.txt", dtype=str, delimiter="\n")
    neut_sent = np.loadtxt("../../data/neut_sentences.txt", dtype=str, delimiter="\n")
    print(male_sent.shape, female_sent.shape, neut_sent.shape)

    male_sent = np.random.choice(male_sent, sample, replace=False)
    female_sent = np.random.choice(female_sent, sample, replace=False)
    neut_sent = np.random.choice(neut_sent, sample, replace=False)
    print(male_sent.shape, female_sent.shape, neut_sent.shape)

    return male_sent, female_sent, neut_sent


def extract_feat_of_context(male_sent, female_sent, neut_sent):
    # obtain features of the last token
    male_feat, female_feat, neut_feat = [], [], []
    with torch.no_grad():
        for sent in male_sent.tolist():
            input_ids = tokenizer.encode(sent, add_special_tokens=False, return_tensors="pt")
            outputs = model.transformer(input_ids=input_ids)[0][0][-1].detach().numpy()    # (2, batch, len, dim)
            male_feat.append(outputs)
        for sent in female_sent.tolist():
            input_ids = tokenizer.encode(sent, add_special_tokens=False, return_tensors="pt")
            outputs = model.transformer(input_ids=input_ids)[0][0][-1].detach().numpy()    # (batch, len, dim)
            female_feat.append(outputs)
        for sent in neut_sent.tolist():
            input_ids = tokenizer.encode(sent, add_special_tokens=False, return_tensors="pt")
            outputs = model.transformer(input_ids=input_ids)[0][0][-1].detach().numpy()    # (batch, len, dim)
            neut_feat.append(outputs)

    # save features into npy files
    male_feat, female_feat, neut_feat = np.array(male_feat), np.array(female_feat), np.array(neut_feat)
    np.save("../../data/male_sentence_clip_feat_random.npy", male_feat)
    np.save("../../data/female_sentence_clip_feat_random.npy", female_feat)
    np.save("../../data/neut_sentence_clip_feat_random.npy", neut_feat)


def split_dataset(male_feat, female_feat, neut_feat):
    # random.seed(0)
    # np.random.seed(0)

    X = np.concatenate((male_feat, female_feat, neut_feat), axis=0)
    # np.random.shuffle(X)
    # X = (X - np.mean(X, axis = 0, keepdims = True)) / np.std(X, axis = 0)
    y_masc = np.ones(male_feat.shape[0], dtype=int)
    y_fem = np.zeros(female_feat.shape[0], dtype=int)
    y_neut = -np.ones(neut_feat.shape[0], dtype=int)
    # y = np.concatenate((masc_scores, fem_scores, neut_scores))#np.concatenate((y_masc, y_fem))
    y = np.concatenate((y_masc, y_fem, y_neut))
    X_train_dev, X_test, y_train_dev, Y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    X_train, X_dev, Y_train, Y_dev = sklearn.model_selection.train_test_split(X_train_dev, y_train_dev, test_size=0.3, random_state=0)
    print("Train size: {}; Dev size: {}; Test size: {}".format(X_train.shape[0], X_dev.shape[0], X_test.shape[0]))

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def train_simple_classifier(X_train, X_dev, X_test, Y_train, Y_dev, Y_test, cls_type="SVM"):
    # simple test how easy this dataset can be classified
    if cls_type == "SVM":
        from sklearn.svm import LinearSVC
        clf = LinearSVC(**{'fit_intercept': False, 'class_weight': None, "dual": False, 'random_state': 0})
        clf.fit(X_train, Y_train)
        count = 0
        count_cls = [0, 0, 0]
        print(X_test.shape)
        Y_pred = clf.predict(X_test)
        print(Y_pred.shape, Y_test.shape)
        for i in range(X_test.shape[0]):
            if Y_pred[i] == Y_test[i]:
                count += 1
                count_cls[Y_pred[i] + 1] += 1
            else:
                pass
        #         print(Y_test[i], Y_pred[i])
        print(count * 1.0 / X_test.shape[0])
        print([x * 3 / X_test.shape[0] for x in count_cls])
    elif cls_type == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier()
        clf.fit(X_train, Y_train)
        count = 0
        Y_pred = clf.predict(X_test)
        print(Y_pred.shape, Y_test.shape)
        for i in range(X_test.shape[0]):
            if Y_pred[i] == Y_test[i]:
                count += 1
        print(count*1.0/X_test.shape[0])
    elif cls_type == "lr":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(penalty='l2')
        clf.fit(X_train, Y_train)
        count = 0
        print(X_test.shape)
        Y_pred = clf.predict(X_test)
        print(Y_pred.shape, Y_test.shape)
        for i in range(X_test.shape[0]):
            if Y_pred[i] == Y_test[i]:
                count += 1
            else:
                pass
        # print(Y_test[i], Y_pred[i])
        print(count*1.0/X_test.shape[0])


def apply_nullspace_projection(X_train, X_dev, X_test, Y_train, Y_dev, Y_test):
    gender_clf = LinearSVC
    # gender_clf = SGDClassifier
    # gender_clf = LogisticRegression
    # gender_clf = LinearDiscriminantAnalysis
    # gender_clf = Perceptron

    params_svc = {'fit_intercept': False, 'class_weight': None, "dual": False, 'random_state': 0}
    params_sgd = {'fit_intercept': False, 'class_weight': None, 'max_iter': 1000, 'random_state': 0}
    params = params_svc
    # params = {'loss': 'hinge', 'n_jobs': 16, 'penalty': 'l2', 'max_iter': 2500, 'random_state': 0}
    # params = {}
    n = 80
    min_acc = 0
    is_autoregressive = True
    dropout_rate = 0

    P, rowspace_projs, Ws = debias.get_debiasing_projection(gender_clf, params, n, 768, is_autoregressive, min_acc,
                                                            X_train, Y_train, X_dev, Y_dev,
                                                            Y_train_main=None, Y_dev_main=None,
                                                            by_class=False, dropout_rate=dropout_rate)
    # np.save("../../data/saved_P/P.npy", P)

    return P, rowspace_projs, Ws


def debias_effect_analysis(P, rowspace_projs, Ws, X_train, X_dev, X_test, Y_train, Y_dev, Y_test):
    def tsne(vecs, labels, title="", ind2label=None, words=None, metric="l2"):
        tsne = TSNE(n_components=2)  # , angle = 0.5, perplexity = 20)   ,  n_iter=3000
        vecs_2d = tsne.fit_transform(vecs)
        label_names = sorted(list(set(labels.tolist())))
        num_labels = len(label_names)

        names = sorted(set(labels.tolist()))

        plt.figure(figsize=(6, 5))
        colors = "red", "blue"
        for i, c, label in zip(sorted(set(labels.tolist())), colors, names):
            plt.scatter(vecs_2d[labels == i, 0], vecs_2d[labels == i, 1], c=c,
                        label=label if ind2label is None else ind2label[label], alpha=0.3,
                        marker="s" if i == 0 else "o")
            plt.legend(loc="upper right")

        plt.title(title)
        plt.savefig("embeddings.{}.png".format(title), dpi=600)
        plt.show()
        return vecs_2d

    all_significantly_biased_vecs = np.concatenate((male_feat, female_feat))
    all_significantly_biased_labels = np.concatenate(
        (np.ones(male_feat.shape[0], dtype=int), np.zeros(female_feat.shape[0], dtype=int)))
    ind2label = {1: "Male-biased", 0: "Female-biased"}
    tsne_before = tsne(all_significantly_biased_vecs, all_significantly_biased_labels, title="Original (t=0)",
                       ind2label=ind2label)

    all_significantly_biased_cleaned = P.dot(all_significantly_biased_vecs.T).T
    tsne_after = tsne(all_significantly_biased_cleaned, all_significantly_biased_labels,
                      title="Projected (t={})".format(n), ind2label=ind2label)

    def perform_purity_test(vecs, k, labels_true):
        np.random.seed(0)
        clustering = sklearn.cluster.KMeans(n_clusters=k)
        clustering.fit(vecs)
        labels_pred = clustering.labels_
        score = sklearn.metrics.homogeneity_score(labels_true, labels_pred)
        return score

    def compute_v_measure(vecs, labels_true, k=2):
        np.random.seed(0)
        clustering = sklearn.cluster.KMeans(n_clusters=k)
        clustering.fit(vecs)
        labels_pred = clustering.labels_
        return sklearn.metrics.v_measure_score(labels_true, labels_pred)

    # remove neutral class, keep only male and female biased
    X_dev = X_dev[Y_dev != -1]
    X_train = X_train[Y_train != -1]
    X_test = X_test[Y_test != -1]

    Y_dev = Y_dev[Y_dev != -1]
    Y_train = Y_train[Y_train != -1]
    Y_test = Y_test[Y_test != -1]

    X_dev_cleaned = (P.dot(X_dev.T)).T
    X_test_cleaned = (P.dot(X_test.T)).T
    X_trained_cleaned = (P.dot(X_train.T)).T

    print("V-measure-before (TSNE space): {}".format(compute_v_measure(tsne_before, all_significantly_biased_labels)))
    print("V-measure-after (TSNE space): {}".format(compute_v_measure(tsne_after, all_significantly_biased_labels)))

    print("V-measure-before (original space): {}".format(
        compute_v_measure(all_significantly_biased_vecs, all_significantly_biased_labels), k=2))
    print("V-measure-after (original space): {}".format(compute_v_measure(X_test_cleaned, Y_test), k=2))

    rank_before = np.linalg.matrix_rank(X_train)
    rank_after = np.linalg.matrix_rank(X_trained_cleaned)
    print("Rank before: {}; Rank after: {}".format(rank_before, rank_after))


if __name__ == '__main__':
    male_sent, female_sent, neut_sent = load_data()
    # extract_feat_of_context(male_sent, female_sent, neut_sent)

    male_feat, female_feat, neut_feat = np.load("../../data/male_sentence_clip_feat_random.npy"), \
                                        np.load("../../data/female_sentence_clip_feat_random.npy"), \
                                        np.load("../../data/neut_sentence_clip_feat_random.npy")

    X_train, X_dev, X_test, Y_train, Y_dev, Y_test = split_dataset(male_feat, female_feat, neut_feat)
    # train_simple_classifier(X_train, X_dev, X_test, Y_train, Y_dev, Y_test, "SVM")

    P, rowspace_projs, Ws = apply_nullspace_projection(X_train, X_dev, X_test, Y_train, Y_dev, Y_test)
    # debias_effect_analysis(P, rowspace_projs, Ws, X_train, X_dev, X_test, Y_train, Y_dev, Y_test)
