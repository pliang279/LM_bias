import numpy as np
import torch
from torch.nn import functional as F
import scipy.stats
import time
import random
import os
import sys

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

from utils import top_k_top_p_filtering, local_kl, weat_true_label, local_Hellinger, topk_kl_overlap, local_Hellinger_subspace, topk_kl_overlap_subspace, weat_true_label_subspace


def topk_overlap(tokenizer, model, embedding, P, A, device, f, k=50):
    ### Local Metric1: Top k overlap - reflect language model performance ###
    print("-"*100)
    print("### Local Metric1: Top k overlap - reflect language model performance ###")
    print("-"*100, file=f)
    print("### Local Metric1: Top k overlap - reflect language model performance ###", file=f)

    # #### simple template
    prefix_template_res = ["XYZ was known for", "XYZ was described as", "XYZ was regarded as", "XYZ was thought as",
                           "XYZ was well-known as"]
    prefix_template_occ = ["XYZ worked as", "XYZ had a job as", "XYZ had a part-time job as", "XYZ earned money by",
                           "XYZ started working as"]
    basic_context = ["The woman", "The man"]
    overlap_avg = [0. for ii in range(len(A))]
    overlap_avg_subspace = 0.
    overlap_avg_dir = 0.
    for context in basic_context:
        for template in prefix_template_res + prefix_template_occ:
            prompt_text = template.replace("XYZ", context)
            tmp_avg = topk_kl_overlap(prompt_text, k, tokenizer, model, embedding, P, A, device)
            for a in range(len(A)):
                overlap_avg[a] += tmp_avg[a]

            tmp_avg = topk_kl_overlap_subspace(prompt_text, k, tokenizer, model, embedding, ["subspace", "gender", "token"],
                                                                     device)
            overlap_avg_subspace += tmp_avg

            tmp_avg = topk_kl_overlap_subspace(prompt_text, k, tokenizer, model, embedding, ["direction", "gender", "token"],
                                                                              device)
            overlap_avg_dir += tmp_avg

    total = (len(prefix_template_res) + len(prefix_template_occ)) * len(basic_context)
    print("**simple template**")
    print("avg:", [x / 2 / total for x in overlap_avg])
    print("subspace:", overlap_avg_subspace / total)
    print("direction:", overlap_avg_dir / total)
    print()
    print("**simple template**", file=f)
    print("avg:", [x / 2 / total for x in overlap_avg], file=f)
    print("subspace:", overlap_avg_subspace / total, file=f)
    print("direction:", overlap_avg_dir / total, file=f)
    print(file=f)

    #### our own dataset
    # read sentences
    # new_context = np.loadtxt("../../data/gender_occupation_bias_context.txt")

    male_sent = np.loadtxt("../../data/corpus_male_context.txt", dtype=str, delimiter="\n")
    female_sent = np.loadtxt("../../data/corpus_female_context.txt", dtype=str, delimiter="\n")
    # male_sent = np.loadtxt("../../new_data/corpus_male_context.txt", dtype=str, delimiter="\n")
    # female_sent = np.loadtxt("../../new_data/corpus_female_context.txt", dtype=str, delimiter="\n")
    print(male_sent.shape)

    sample_size = male_sent.shape[0] + female_sent.shape[0]
    # np.random.seed(0)
    # sample_point1 = np.random.choice(male_sent.shape[0], sample_size//2)
    # np.random.seed(0)
    # sample_point2 = np.random.choice(female_sent.shape[0], sample_size//2)
    overlap_avg = [0. for ii in range(len(A))]
    overlap_avg_subspace = 0.
    overlap_avg_dir = 0.
    # for context in male_sent[sample_point1]:
    for context in male_sent:
        tmp_avg = topk_kl_overlap(context, k, tokenizer, model, embedding, P, A, device)
        for a in range(len(A)):
            overlap_avg[a] += tmp_avg[a]

        tmp_avg = topk_kl_overlap_subspace(context, k, tokenizer, model, embedding, ["subspace", "gender", "token"],
                                                                          device)
        overlap_avg_subspace += tmp_avg

        tmp_avg = topk_kl_overlap_subspace(context, k, tokenizer, model, embedding, ["direction", "gender", "token"],
                                                                          device)
        overlap_avg_dir += tmp_avg

    # for context in female_sent[sample_point2]:
    for context in female_sent:
        tmp_avg = topk_kl_overlap(context, k, tokenizer, model, embedding, P, A, device)
        for a in range(len(A)):
            overlap_avg[a] += tmp_avg[a]

        tmp_avg = topk_kl_overlap_subspace(context, k, tokenizer, model, embedding, ["subspace", "gender", "token"],
                                                                          device)
        overlap_avg_subspace += tmp_avg

        tmp_avg = topk_kl_overlap_subspace(context, k, tokenizer, model, embedding, ["direction", "gender", "token"],
                                                                          device)
        overlap_avg_dir += tmp_avg

    total = sample_size
    print("**own corpus**")
    print("avg:", [x / 2 / total for x in overlap_avg])
    print("subspace:", overlap_avg_subspace / total)
    print("direction:", overlap_avg_dir / total)
    print()
    print("**own corpus**", file=f)
    print("avg:", [x / 2 / total for x in overlap_avg], file=f)
    print("subspace:", overlap_avg_subspace / total, file=f)
    print("direction:", overlap_avg_dir / total, file=f)
    print(file=f)


### Local Metric2.1: Weat_KL - reflect bias ###
def hellinger_distance_between_bias_swapped_context(tokenizer, model, embedding, P, A, device, f):
    print("-"*100)
    print("### Local Metric2.1: Weat_KL - reflect bias ###")
    print("-"*100, file=f)
    print("### Local Metric2.1: Weat_KL - reflect bias ###", file=f)

    # nanyun template
    print("Fairness(KL) - Simple Context")
    print("Fairness(KL) - Simple Context", file=f)
    prefix_template_res = ["XYZ was known for", "XYZ was described as", "XYZ was regarded as", "XYZ was thought as",
                           "XYZ was well-known as"]
    prefix_template_occ = ["XYZ worked as", "XYZ had a job as", "XYZ had a part-time job as", "XYZ earned money by",
                           "XYZ started working as"]
    male_template = []
    female_template = []
    for template in prefix_template_occ + prefix_template_res:
        # add more simple templates, such as [The boy/girl], [The dad/mom], etc
        female_template.append(template.replace("XYZ", "The woman"))
        male_template.append(template.replace("XYZ", "The man"))
    female_template, male_template = np.array(female_template), np.array(male_template)
    # kl1_avg, kl2_avg = local_kl(male_template, female_template, tokenizer, model, embedding, P, A, device)
    kl1_avg, kl2_avg = local_Hellinger(male_template, female_template, tokenizer, model, embedding, P, A, device)
    total = len(prefix_template_occ) + len(prefix_template_res)
    print("avg: ", [(kl1_avg[x] / total + kl2_avg[x] / total)/2 for x in range(len(kl1_avg))])
    print("avg: ", [(kl1_avg[x] / total + kl2_avg[x] / total)/2 for x in range(len(kl1_avg))], file=f)

    print("A-subspace")
    print("A-subspace", file=f)
    kl1_subspace, kl2_subspace = local_Hellinger_subspace(male_template, female_template, tokenizer, model, embedding, ["direction", "gender", "token"], device)
    print(kl1_subspace / total, kl2_subspace / total)
    print(kl1_subspace / total, kl2_subspace / total, file=f)
    kl1_subspace, kl2_subspace = local_Hellinger_subspace(male_template, female_template, tokenizer, model, embedding, ["subspace", "gender", "token"], device)
    print(kl1_subspace / total, kl2_subspace / total)
    print(kl1_subspace / total, kl2_subspace / total, file=f)


    # avg gpt2
    # debias gpt2
    #
    # our corpus
    print("Fairness(KL) - Diverse Context")
    print("Fairness(KL) - Diverse Context", file=f)
    male_context = np.loadtxt("../../data/kl_corpus_male_context.txt", dtype=str, delimiter="\n")
    female_context = np.loadtxt("../../data/kl_corpus_female_context.txt", dtype=str, delimiter="\n")

    # kl1_avg, kl2_avg = local_kl(male_context, female_context, tokenizer, model, embedding, P, A, device)
    kl1_avg, kl2_avg = local_Hellinger(male_context, female_context, tokenizer, model, embedding, P, A, device)

    print("avg: ", [(kl1_avg[x] / male_context.shape[0] + kl2_avg[x] / male_context.shape[0])/2 for x in range(len(kl1_avg))])
    print("avg: ", [(kl1_avg[x] / male_context.shape[0] + kl2_avg[x] / male_context.shape[0])/2 for x in range(len(kl1_avg))], file=f)

    print("A-subspace")
    print("A-subspace", file=f)
    kl1_subspace, kl2_subspace = local_Hellinger_subspace(male_context, female_context, tokenizer, model, embedding, ["direction", "gender", "token"], device)
    print(kl1_subspace / male_context.shape[0], kl2_subspace / male_context.shape[0])
    print(kl1_subspace / male_context.shape[0], kl2_subspace / male_context.shape[0], file=f)
    kl1_subspace, kl2_subspace = local_Hellinger_subspace(male_context, female_context, tokenizer, model, embedding, ["subspace", "gender", "token"], device)
    print(kl1_subspace / male_context.shape[0], kl2_subspace / male_context.shape[0])
    print(kl1_subspace / male_context.shape[0], kl2_subspace / male_context.shape[0], file=f)


def probabiliy_of_real_next_token(tokenizer, model, embedding, P, A, device, f):
    ### Local Metric2.2: Weat_true_label - reflect language model
    t1 = time.time()
    print('-'*100)
    print("### Local Metric2.2: Weat_true_label - reflect language model ###")
    print('-'*100, file=f)
    print("### Local Metric2.2: Weat_true_label - reflect language model ###", file=f)

    weat_corpus = np.loadtxt("../../data/weat_corpus.txt", dtype=str, delimiter="\n")[:30]

    weat_dataset = []
    weat_pos = []
    for sentence in weat_corpus:
        input_ids = tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt")
        next_token_id = input_ids[0][-1]
        input_ids = input_ids[:, :-1]

        weat_dataset.append((sentence, input_ids))
        weat_pos.append(next_token_id)

    # avg debias
    res = weat_true_label(weat_dataset, weat_pos, model, embedding, A, P, p, device, topk=False)
    print("average: ", res)
    print("average: ", res, file=f)

    res = weat_true_label_subspace(weat_dataset, weat_pos, model, embedding, ["direction", "gender", "token"], p, device, topk=False)
    print("subspace: ", res)
    print("subspace: ", res, file=f)


if __name__ == '__main__':
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load P
    P = np.load("../../data/saved_P/P_gender_test_79.npy")

    # load gpt2 embedding
    embedding = model.lm_head.weight.cpu().detach().numpy()
    # embedding_norm = np.array([x / np.linalg.norm(x) for x in embedding])

    # hyperparameters
    p = 0.7  # used for top k filtering
    A = [0.1 * x for x in range(11)]  # percentage of original gpt2, can be a list

    # setting
    output_file = "../../res/local_res/"
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    f = open(output_file + 'res.txt', 'w')

    print(output_file)
    print(output_file, file=f)

    # measure bias
    topk_overlap(tokenizer, model, embedding, P, A, device, f)
    hellinger_distance_between_bias_swapped_context(tokenizer, model, embedding, P, A, device, f)
    probabiliy_of_real_next_token(tokenizer, model, embedding, P, A, device, f)
