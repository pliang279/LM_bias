import torch
import numpy as np
from torch.nn import functional as F
import scipy.stats
from sklearn.decomposition import PCA
import json


def doPCA(pairs, num_components=10):
    matrix = []
    for a, b in pairs:
        center = (a + b) / 2
        norm_a = a - center
        norm_b = b - center
        norm_a, norm_b = norm_a.detach().numpy(), norm_b.detach().numpy()
        # norm_a, norm_b = norm_a/np.linalg.norm(norm_a), norm_b/np.linalg.norm(norm_b)
        matrix.append(norm_a)
        matrix.append(norm_b)
    matrix = np.array(matrix)
    pca = PCA(n_components=num_components, svd_solver="full")
    pca.fit(matrix)  # Produce different results each time...
    return pca

def dropspace(u, V):
    # u, V = u.detach().numpy(), V.detach().numpy()
    norm_sqrd = np.sum(V*V, axis=-1)
    vecs = np.divide(V@u, norm_sqrd)[:, None] * V
    subspace = np.sum(vecs, axis=0)
    return u - subspace

def drop_bias(u, v):
    # return u - torch.ger(torch.matmul(u, v), v) / v.dot(v)
    projection = u.dot(v) * v / np.linalg.norm(v)
    return u - projection

def drop(u, v):
    return u - v * u.dot(v) / v.dot(v)

def top_k_top_p_filtering(
    logits,    # (1, 50257)
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def topk_kl_overlap(prompt_text, k, tokenizer, model, embedding, P, A, device):
    """
        :param prompt_text: a single prompt
        :param k: top k
        :param tokenizer: tokenizer
        :param model: gpt2 or other language model
        :param embedding: gpt2 word embedding
        :param P: nullspace matrix
        :param A: alpha list
        :param device: cpu or gpu
        """
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    # original gpt2 model
    input_ids = input_ids.to(device)
    outputs = model.transformer(input_ids=input_ids)[0][0][-1]  # (2, batch, len, dim)
    outputs = outputs.cpu().detach().numpy()
    logits = embedding.dot(outputs)

    old_rank = np.argsort(-logits).tolist()
    old_logits = np.sort(-logits).tolist()
    topk_raw = old_rank[:k]
    logits_raw = [-x for x in old_logits[:k]]

    # averaged hidden state debiased gpt2 model
    outputs_P = P.dot(outputs.T).T
    KL1 = [0 for ii in range(len(A))]
    KL2 = [0 for ii in range(len(A))]
    for a in range(len(A)):
        avg_outputs = A[a] * outputs + (1 - A[a]) * outputs_P
        avg_logits = embedding.dot(avg_outputs)

        logits_new = []
        for i, token in enumerate(topk_raw):
            logits_new.append(avg_logits[token])
        logits_new = np.array(logits_new)

        KL1[a] = scipy.stats.entropy(logits_raw, logits_new)
        KL2[a] = scipy.stats.entropy(logits_new, logits_raw)

    return KL1 + KL2


def topk_kl_overlap_subspace(prompt_text, k, tokenizer, model, embedding, mode, device):
    if mode[1] == "gender":
        if mode[0] == "direction":
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_direction.npy")
            debiased_embedding = np.array([drop(embedding[i], gender_direction) for i in range(embedding.shape[0])])
        else:
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_subspace.npy")
            debiased_embedding = np.array([dropspace(embedding[i], gender_direction) for i in range(embedding.shape[0])])
    else:
        religion_dir1 = np.load("../../data/bias_subspace/religion_direction1.npy")
        religion_dir2 = np.load("../../data/bias_subspace/religion_direction2.npy")
        religion_dir3 = np.load("../../data/bias_subspace/religion_direction3.npy")
        debiased_embedding = np.array([drop(embedding[i], religion_dir1) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir2) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir3) for i in range(embedding.shape[0])])

    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    # original gpt2 model
    input_ids = input_ids.to(device)
    outputs = model.transformer(input_ids=input_ids)[0][0][-1]  # (2, batch, len, dim)
    outputs = outputs.cpu().detach().numpy()
    logits = embedding.dot(outputs)

    old_rank = np.argsort(-logits).tolist()
    old_logits = np.sort(-logits).tolist()
    topk_raw = old_rank[:k]
    logits_raw = [-x for x in old_logits[:k]]

    # averaged hidden state debiased gpt2 model
    avg_outputs = outputs
    avg_logits = debiased_embedding.dot(avg_outputs)

    logits_new = []
    for i, token in enumerate(topk_raw):
        logits_new.append(avg_logits[token])
    logits_new = np.array(logits_new)

    KL1 = scipy.stats.entropy(logits_raw, logits_new)
    KL2 = scipy.stats.entropy(logits_new, logits_raw)

    return (KL1 + KL2) / 2


def local_kl(male_context, female_context, tokenizer, model, embedding, P, A, device):
    kl1_avg = [0. for ii in range(len(A))]
    kl2_avg = [0. for ii in range(len(A))]
    for i in range(male_context.shape[0]):
        input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_m = input_ids_m.to(device)
        outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        outputs_P = P.dot(outputs.T).T

        input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_f = input_ids_f.to(device)
        outputs_f = model.transformer(input_ids=input_ids_f)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        outputs_P_f = P.dot(outputs_f.T).T

        for a in range(len(A)):
            outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
            new_logits = embedding.dot(outputs_P)
            new_logits = torch.from_numpy(new_logits).float()
            new_logits = new_logits.unsqueeze(0)  # [1, 50257]
            probs_m = F.softmax(new_logits, dim=-1)

            outputs_P_f = (1 - A[a]) * outputs_P_f + A[a] * outputs_f
            new_logits_f = embedding.dot(outputs_P_f)
            new_logits_f = torch.from_numpy(new_logits_f).float()
            new_logits_f = new_logits_f.unsqueeze(0)  # [1, 50257]
            probs_f = F.softmax(new_logits_f, dim=-1)

            KL1 = scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy())
            KL2 = scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy())

            kl1_avg[a] += KL1
            kl2_avg[a] += KL2

    return kl1_avg, kl2_avg


def local_Hellinger(male_context, female_context, tokenizer, model, embedding, P, A, device):
    kl1_avg = [0. for ii in range(len(A))]
    kl2_avg = [0. for ii in range(len(A))]
    for i in range(male_context.shape[0]):
        input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_m = input_ids_m.to(device)
        outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim), embedding for male context
        outputs_P = P.dot(outputs.T).T      # debiased embedding for male context

        input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_f = input_ids_f.to(device)
        outputs_f = model.transformer(input_ids=input_ids_f)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim), embedding for female context
        outputs_P_f = P.dot(outputs_f.T).T      # debiased embedding for female context

        for a in range(len(A)):
            outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
            new_logits = embedding.dot(outputs_P)
            new_logits = torch.from_numpy(new_logits).float()
            new_logits = new_logits.unsqueeze(0)  # [1, 50257]
            probs_m = F.softmax(new_logits, dim=-1)

            outputs_P_f = (1 - A[a]) * outputs_P_f + A[a] * outputs_f
            new_logits_f = embedding.dot(outputs_P_f)
            new_logits_f = torch.from_numpy(new_logits_f).float()
            new_logits_f = new_logits_f.unsqueeze(0)  # [1, 50257]
            probs_f = F.softmax(new_logits_f, dim=-1)

            hell1 = np.sqrt(1-np.sum(np.sqrt(probs_m[0].detach().numpy()*probs_f[0].detach().numpy())))
            hell2 = np.sqrt(1-np.sum(np.sqrt(probs_f[0].detach().numpy()*probs_m[0].detach().numpy())))
            # KL1 = scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy())
            # KL2 = scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy())

            kl1_avg[a] += hell1
            kl2_avg[a] += hell2

            # print(hell1)

    return kl1_avg, kl2_avg


def local_Hellinger_sensitive(male_context, female_context, tokenizer, model, embedding, P, device):
    stop_word = np.loadtxt(open("../../data/stopword.list", "r"), dtype='str')
    stop_word = set(x for x in stop_word)
    with open("../../data/glove_religion_similarity.json") as ff:
        similarity = json.load(ff)
    for w in stop_word:
        similarity['judaism'][w] = 0
        similarity['christianity'][w] = 0
        similarity['islam'][w] = 0
    for w in ["al", "lacking", "lack", "countries", "country", "government", "nation", "cyber", "translator",
              "journalist", "situation", "early"]:
        similarity['judaism'][w] = 0
        similarity['christianity'][w] = 0
        similarity['islam'][w] = 0
    bias_thre = (0.16, 0.15, 0.17)

    kl1_avg = 0.
    kl2_avg = 0.
    for i in range(male_context.shape[0]):
        input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_m = input_ids_m.to(device)
        # sensitive ----
        model_inputs = model.prepare_inputs_for_generation(input_ids_m, past=None, attention_mask=input_ids_m.new_ones(input_ids_m.shape),
                                                           use_cache=True)

        outputs = model(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]  # batch * vocab
        logits_filter = top_k_top_p_filtering(next_token_logits, top_p=0.7)  # batch * vocab
        top_p_mask = logits_filter.eq(-float("Inf"))

        top_k_tokens = []
        ii = 0
        tmp = (top_p_mask == False)[ii].nonzero().cpu().detach().numpy().tolist()  # batch tuple
        top_k_tokens.append([int(x) for x in torch.sort(-logits_filter[ii])[1][:len(tmp)]])
        probs_bias = F.softmax(logits_filter, dim=-1).cpu().detach().numpy()  # batch * vocab

        normalize = 0
        jew_token, christian_token, muslim_token = [], [], []
        jew_prob, christian_prob, muslim_prob = 0, 0, 0
        token_num = 0
        for t in top_k_tokens[ii]:
            glove_token = tokenizer.decoder[int(t)]
            if glove_token[0] != "Ġ":
                token_num += 1
                normalize += probs_bias[ii][int(t)]
                continue
            glove_token = glove_token[1:]
            if len(glove_token) <= 1:
                continue
            if glove_token[0] >= "A" and glove_token[0] <= "Z":
                token_num += 1
                normalize += probs_bias[ii][int(t)]
                continue
            # glove_token = glove_token.lower()
            if glove_token in similarity["judaism"].keys():  # and glove_token not in stop_word:
                if similarity["judaism"][glove_token] > bias_thre[0]:
                    jew_token.append(tokenizer.decoder[int(t)])
                    jew_prob += probs_bias[ii][int(t)]
                if similarity["christianity"][glove_token] > bias_thre[1]:
                    christian_token.append(tokenizer.decoder[int(t)])
                    christian_prob += probs_bias[ii][int(t)]
                if similarity["islam"][glove_token] > bias_thre[2]:
                    muslim_token.append(tokenizer.decoder[int(t)])
                    muslim_prob += probs_bias[ii][int(t)]
                token_num += 1
                normalize += probs_bias[ii][int(t)]
        if normalize == 0:
            normalize = 1
        if token_num == 0:
            token_num = 1
        jew_prob, christian_prob, muslim_prob = jew_prob / normalize, christian_prob / normalize, muslim_prob / normalize
        if jew_prob >= 0.1 or muslim_prob >= 0.1 or christian_prob >= 0.1 or len(
                    jew_token) / token_num >= 0.3 or len(muslim_token) / token_num >= 0.3 or len(
                    christian_token) / token_num >= 0.3:  # or tokenizer.decode(input_ids[0]) == "The muslim was described as being":
                # if bias1 >= bias_thre[0] or bias2 >= bias_thre[1] or bias3 >= bias_thre[2]:
                #     ratio[ii] = max(1 - abs(max(bias1, bias2, bias3)), 0.6)
            A_m = 0.7
        else:
            A_m = 1

        # ---------

        input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_f = input_ids_f.to(device)

        # sensitive ----
        model_inputs = model.prepare_inputs_for_generation(input_ids_f, past=None,
                                                           attention_mask=input_ids_f.new_ones(input_ids_f.shape),
                                                           use_cache=True)

        outputs = model(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]  # batch * vocab
        logits_filter = top_k_top_p_filtering(next_token_logits, top_p=0.7)  # batch * vocab
        top_p_mask = logits_filter.eq(-float("Inf"))

        top_k_tokens = []
        ii = 0
        tmp = (top_p_mask == False)[ii].nonzero().cpu().detach().numpy().tolist()  # batch tuple
        top_k_tokens.append([int(x) for x in torch.sort(-logits_filter[ii])[1][:len(tmp)]])
        probs_bias = F.softmax(logits_filter, dim=-1).cpu().detach().numpy()  # batch * vocab

        normalize = 0
        jew_token, christian_token, muslim_token = [], [], []
        jew_prob, christian_prob, muslim_prob = 0, 0, 0
        token_num = 0
        for t in top_k_tokens[ii]:
            glove_token = tokenizer.decoder[int(t)]
            if glove_token[0] != "Ġ":
                token_num += 1
                normalize += probs_bias[ii][int(t)]
                continue
            glove_token = glove_token[1:]
            if len(glove_token) <= 1:
                continue
            if glove_token[0] >= "A" and glove_token[0] <= "Z":
                token_num += 1
                normalize += probs_bias[ii][int(t)]
                continue
            # glove_token = glove_token.lower()
            if glove_token in similarity["judaism"].keys():  # and glove_token not in stop_word:
                if similarity["judaism"][glove_token] > bias_thre[0]:
                    jew_token.append(tokenizer.decoder[int(t)])
                    jew_prob += probs_bias[ii][int(t)]
                if similarity["christianity"][glove_token] > bias_thre[1]:
                    christian_token.append(tokenizer.decoder[int(t)])
                    christian_prob += probs_bias[ii][int(t)]
                if similarity["islam"][glove_token] > bias_thre[2]:
                    muslim_token.append(tokenizer.decoder[int(t)])
                    muslim_prob += probs_bias[ii][int(t)]
                token_num += 1
                normalize += probs_bias[ii][int(t)]
        if normalize == 0:
            normalize = 1
        if token_num == 0:
            token_num = 1
        jew_prob, christian_prob, muslim_prob = jew_prob / normalize, christian_prob / normalize, muslim_prob / normalize
        if jew_prob >= 0.1 or muslim_prob >= 0.1 or christian_prob >= 0.1 or len(
                jew_token) / token_num >= 0.3 or len(muslim_token) / token_num >= 0.3 or len(
            christian_token) / token_num >= 0.3:  # or tokenizer.decode(input_ids[0]) == "The muslim was described as being":
            # if bias1 >= bias_thre[0] or bias2 >= bias_thre[1] or bias3 >= bias_thre[2]:
            #     ratio[ii] = max(1 - abs(max(bias1, bias2, bias3)), 0.6)
            A_f = 0.7
        else:
            A_f = 1
        print(A_f, A_m)

        # ---------

        outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        outputs_P = P.dot(outputs.T).T

        outputs_f = model.transformer(input_ids=input_ids_f)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        outputs_P_f = P.dot(outputs_f.T).T

        outputs_P = (1 - A_m) * outputs_P + A_m * outputs
        new_logits = embedding.dot(outputs_P)
        new_logits = torch.from_numpy(new_logits).float()
        new_logits = new_logits.unsqueeze(0)  # [1, 50257]
        probs_m = F.softmax(new_logits, dim=-1)

        outputs_P_f = (1 - A_f) * outputs_P_f + A_f * outputs_f
        new_logits_f = embedding.dot(outputs_P_f)
        new_logits_f = torch.from_numpy(new_logits_f).float()
        new_logits_f = new_logits_f.unsqueeze(0)  # [1, 50257]
        probs_f = F.softmax(new_logits_f, dim=-1)

        hell1 = np.sqrt(1-np.sum(np.sqrt(probs_m[0].detach().numpy()*probs_f[0].detach().numpy())))
        hell2 = np.sqrt(1-np.sum(np.sqrt(probs_f[0].detach().numpy()*probs_m[0].detach().numpy())))
        # KL1 = scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy())
        # KL2 = scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy())

        kl1_avg += hell1
        kl2_avg += hell2

    return kl1_avg, kl2_avg


def local_Hellinger_subspace(male_context, female_context, tokenizer, model, embedding, mode, device):
    if mode[1] == "gender":
        if mode[0] == "direction":
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_direction.npy")
            debiased_embedding = np.array([drop(embedding[i], gender_direction) for i in range(embedding.shape[0])])
        else:
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_subspace.npy")
            debiased_embedding = np.array([dropspace(embedding[i], gender_direction) for i in range(embedding.shape[0])])
        # self.embedding.to(self.args.device)
    elif mode[1] == "religion":
        religion_dir1 = np.load("../../data/bias_subspace/religion_direction1.npy")
        religion_dir2 = np.load("../../data/bias_subspace/religion_direction2.npy")
        religion_dir3 = np.load("../../data/bias_subspace/religion_direction3.npy")
        debiased_embedding = np.array([drop(embedding[i], religion_dir1) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir2) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir3) for i in range(embedding.shape[0])])

    kl1_avg = 0.
    kl2_avg = 0.
    for i in range(male_context.shape[0]):
        input_ids_m = tokenizer.encode(male_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_m = input_ids_m.to(device)
        outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        # outputs_P = P.dot(outputs.T).T
        new_logits = debiased_embedding.dot(outputs)
        new_logits = torch.from_numpy(new_logits).float()
        new_logits = new_logits.unsqueeze(0)  # [1, 50257]
        probs_m = F.softmax(new_logits, dim=-1)

        input_ids_f = tokenizer.encode(female_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids_f = input_ids_f.to(device)
        outputs_f = model.transformer(input_ids=input_ids_f)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        # outputs_P_f = P.dot(outputs_f.T).T
        new_logits_f = debiased_embedding.dot(outputs_f)
        new_logits_f = torch.from_numpy(new_logits_f).float()
        new_logits_f = new_logits_f.unsqueeze(0)  # [1, 50257]
        probs_f = F.softmax(new_logits_f, dim=-1)

        hell1 = np.sqrt(1-np.sum(np.sqrt(probs_m[0].detach().numpy()*probs_f[0].detach().numpy())))
        hell2 = np.sqrt(1-np.sum(np.sqrt(probs_f[0].detach().numpy()*probs_m[0].detach().numpy())))
        # KL1 = scipy.stats.entropy(probs_m[0].detach().numpy(), probs_f[0].detach().numpy())
        # KL2 = scipy.stats.entropy(probs_f[0].detach().numpy(), probs_m[0].detach().numpy())

        kl1_avg += hell1
        kl2_avg += hell2

    return kl1_avg, kl2_avg


def weat_true_label(weat_dataset, weat_pos, model, embedding, A, P, p, device, topk=False):
    if topk:
        weat_topk = 0.
        count = 0
        for i in range(len(weat_dataset)):
            input_ids_m = weat_dataset[i][1]
            input_ids_m = input_ids_m.to(device)
            outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            logits = embedding.dot(outputs)
            logits_filter = torch.from_numpy(logits).float().clone()
            logits_filter = logits_filter.unsqueeze(0)
            logits_filter = top_k_top_p_filtering(logits_filter, top_p=p)
            top_p_mask = logits_filter.eq(-float("Inf"))

            outputs_P = P.dot(outputs.T).T
            new_logits = embedding.dot(outputs_P)
            logits_filter = torch.from_numpy(new_logits).float().clone()
            logits_filter = logits_filter.unsqueeze(0)
            logits_filter.masked_fill_(top_p_mask, -float("Inf"))
            probs_m = F.softmax(logits_filter, dim=-1).detach().numpy()

            weat_topk += probs_m[0][weat_pos[i]]
            count += 1
        return weat_topk / count

    weat_avg = [0. for ii in range(len(A))]
    count = 0
    for i in range(len(weat_dataset)):
        input_ids_m = weat_dataset[i][1]
        input_ids_m = input_ids_m.to(device)
        outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        outputs_P = P.dot(outputs.T).T
        for a in range(len(A)):
            outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
            new_logits = embedding.dot(outputs_P)
            new_logits = torch.from_numpy(new_logits).float()
            new_logits = new_logits.unsqueeze(0)  # [1, 50257]
            probs_m = F.softmax(new_logits, dim=-1).detach().numpy()

            weat_avg[a] += probs_m[0][weat_pos[i]]
        count += 1
    return [x / count for x in weat_avg]

def weat_true_label_sensitive(weat_dataset, weat_pos, model, embedding, mode, p, device, topk=False):
    if topk:
        weat_topk = 0.
        count = 0
        for i in range(len(weat_dataset)):
            input_ids_m = weat_dataset[i][1]
            input_ids_m = input_ids_m.to(device)
            outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
            logits = embedding.dot(outputs)
            logits_filter = torch.from_numpy(logits).float().clone()
            logits_filter = logits_filter.unsqueeze(0)
            logits_filter = top_k_top_p_filtering(logits_filter, top_p=p)
            top_p_mask = logits_filter.eq(-float("Inf"))

            outputs_P = P.dot(outputs.T).T
            new_logits = embedding.dot(outputs_P)
            logits_filter = torch.from_numpy(new_logits).float().clone()
            logits_filter = logits_filter.unsqueeze(0)
            logits_filter.masked_fill_(top_p_mask, -float("Inf"))
            probs_m = F.softmax(logits_filter, dim=-1).detach().numpy()

            weat_topk += probs_m[0][weat_pos[i]]
            count += 1
        return weat_topk / count

    weat_avg = [0. for ii in range(len(A))]
    count = 0
    for i in range(len(weat_dataset)):
        input_ids_m = weat_dataset[i][1]
        input_ids_m = input_ids_m.to(device)
        outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        outputs_P = P.dot(outputs.T).T
        for a in range(len(A)):
            outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
            new_logits = embedding.dot(outputs_P)
            new_logits = torch.from_numpy(new_logits).float()
            new_logits = new_logits.unsqueeze(0)  # [1, 50257]
            probs_m = F.softmax(new_logits, dim=-1).detach().numpy()

            weat_avg[a] += probs_m[0][weat_pos[i]]
        count += 1
    return [x / count for x in weat_avg]

def weat_true_label_subspace(weat_dataset, weat_pos, model, embedding, mode, p, device, topk=False):
    if mode[1] == "gender":
        if mode[0] == "direction":
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_direction.npy")
            debiased_embedding = np.array([drop(embedding[i], gender_direction) for i in range(embedding.shape[0])])
        else:
            gender_direction = np.load("../../data/bias_subspace/gpt2_gender_subspace.npy")
            debiased_embedding = np.array([dropspace(embedding[i], gender_direction) for i in range(embedding.shape[0])])
        # self.embedding.to(self.args.device)
    elif mode[1] == "religion":
        religion_dir1 = np.load("../../data/bias_subspace/religion_direction1.npy")
        religion_dir2 = np.load("../../data/bias_subspace/religion_direction2.npy")
        religion_dir3 = np.load("../../data/bias_subspace/religion_direction3.npy")
        debiased_embedding = np.array([drop(embedding[i], religion_dir1) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir2) for i in range(embedding.shape[0])])
        debiased_embedding = np.array([drop(debiased_embedding[i], religion_dir3) for i in range(embedding.shape[0])])

    weat_avg = 0.
    count = 0
    for i in range(len(weat_dataset)):
        input_ids_m = weat_dataset[i][1]
        input_ids_m = input_ids_m.to(device)
        outputs = model.transformer(input_ids=input_ids_m)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        # outputs_P = P.dot(outputs.T).T
        # for a in range(len(A)):
        #     outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
        new_logits = debiased_embedding.dot(outputs)
        new_logits = torch.from_numpy(new_logits).float()
        new_logits = new_logits.unsqueeze(0)  # [1, 50257]
        probs_m = F.softmax(new_logits, dim=-1).detach().numpy()

            # weat_avg[a] += probs_m[0][weat_pos[i]]
        weat_avg += probs_m[0][weat_pos[i]]
        count += 1
    return weat_avg / count


def local_kl_reverse(occ_context, tokenizer, model, embedding, pairs_id, A, P, device):
    kl = [0. for ii in range(len(A))]
    for i in range(occ_context.shape[0]):
        input_ids = tokenizer.encode(occ_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(device)
        outputs = model.transformer(input_ids=input_ids)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        outputs_P = P.dot(outputs.T).T

        for a in range(len(A)):
            outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
            new_logits = embedding.dot(outputs_P)
            new_logits = torch.from_numpy(new_logits).float()
            new_logits = new_logits.unsqueeze(0)  # [1, 50257]
            probs = F.softmax(new_logits, dim=-1)
            probs = probs.cpu().detach().numpy()

            occ_prob1 = 0.
            occ_prob2 = 0.
            for p1, p2 in pairs_id:
                occ_prob1 += probs[0][p1]
                occ_prob2 += probs[0][p2]

            tmp_kl1 = 0.
            tmp_kl2 = 0.
            for p1, p2 in pairs_id:
                tmp_kl1 += probs[0][p1]/occ_prob1 * np.log(probs[0][p1] / occ_prob1 / probs[0][p2] * occ_prob2)
                tmp_kl2 += probs[0][p2]/occ_prob2 * np.log(probs[0][p2] / occ_prob2 / probs[0][p1] * occ_prob1)
            kl[a] += (tmp_kl1 + tmp_kl2) / 2

    return kl


def local_kl_reverse_geometry(occ_context, tokenizer, model, embedding, pairs_id, num_components=2, device="cpu"):
    def doPCA(pairs, num_components=10):
        matrix = []
        for a, b in pairs:
            center = (a + b)/2
            norm_a = a - center
            norm_b = b - center
            norm_a, norm_b = norm_a.detach().numpy(), norm_b.detach().numpy()
            # norm_a, norm_b = norm_a/np.linalg.norm(norm_a), norm_b/np.linalg.norm(norm_b)
            matrix.append(norm_a)
            matrix.append(norm_b)
        matrix = np.array(matrix)
        pca = PCA(n_components=num_components, svd_solver="full")
        pca.fit(matrix) # Produce different results each time...
        return pca

    def dropspace(u, V):
        # u, V = u.detach().numpy(), V.detach().numpy()
        norm_sqrd = np.sum(V*V, axis=-1)
        vecs = np.divide(V@u, norm_sqrd)[:, None] * V
        subspace = np.sum(vecs, axis=0)
        return u - subspace

    pairs = []
    for female, male in pairs_id:
        female_feat, male_feat = embedding[female], embedding[male]
        female_feat, male_feat = female_feat/np.linalg.norm(female_feat), male_feat/np.linalg.norm(male_feat)
        if type(male_feat) is np.ndarray:
            female_feat, male_feat = torch.from_numpy(female_feat), torch.from_numpy(male_feat)
        pairs.append((female_feat, male_feat))
    pca_res = doPCA(pairs, num_components=num_components)
    print("pca_res.explained_variance_ratio_: ", pca_res.explained_variance_ratio_)
    print("pca shape", pca_res.components_.shape)
    gender_dir1 = torch.from_numpy(pca_res.components_[0])
    gender_dir2 = torch.from_numpy(pca_res.components_[1])
    # gender_dir = torch.from_numpy(pca_res.components_[:num_components])
    gender_dir = pca_res.components_[:num_components]

    # kl = [0. for ii in range(len(A))]
    kl = 0.
    for i in range(occ_context.shape[0]):
        input_ids = tokenizer.encode(occ_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(device)
        outputs = model.transformer(input_ids=input_ids)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        logits = embedding.dot(outputs)
        logits = torch.from_numpy(logits).float()
        logits = logits.unsqueeze(0)
        probs = F.softmax(logits, dim=-1)
        probs = probs.cpu().detach().numpy()

        occ_prob1 = 0.
        occ_prob2 = 0.
        for p1, p2 in pairs_id:
            occ_prob1 += probs[0][p1]
            occ_prob2 += probs[0][p2]

        tmp_kl1 = 0.
        tmp_kl2 = 0.
        for p1, p2 in pairs_id:
            tmp_kl1 += probs[0][p1] / occ_prob1 * np.log(probs[0][p1] / occ_prob1 / probs[0][p2] * occ_prob2)
            tmp_kl2 += probs[0][p2] / occ_prob2 * np.log(probs[0][p2] / occ_prob2 / probs[0][p1] * occ_prob1)
        kl += (tmp_kl1 + tmp_kl2) / 2

    tmp = model.lm_head.weight.data
    model.lm_head.weight.data = torch.from_numpy(
        np.array([dropspace(embedding[i], gender_dir) for i in range(embedding.shape[0])]))

    kl_debias = 0.
    for i in range(occ_context.shape[0]):
        input_ids = tokenizer.encode(occ_context[i], add_special_tokens=False, return_tensors="pt")
        input_ids = input_ids.to(device)
        outputs = model.transformer(input_ids=input_ids)[0][0][-1].cpu().detach().numpy()  # (2, batch, len, dim)
        logits = embedding.dot(outputs)
        logits = torch.from_numpy(logits).float()
        logits = logits.unsqueeze(0)
        probs = F.softmax(logits, dim=-1)
        probs = probs.cpu().detach().numpy()

        occ_prob1 = 0.
        occ_prob2 = 0.
        for p1, p2 in pairs_id:
            occ_prob1 += probs[0][p1]
            occ_prob2 += probs[0][p2]

        tmp_kl1 = 0.
        tmp_kl2 = 0.
        for p1, p2 in pairs_id:
            tmp_kl1 += probs[0][p1] / occ_prob1 * np.log(probs[0][p1] / occ_prob1 / probs[0][p2] * occ_prob2)
            tmp_kl2 += probs[0][p2] / occ_prob2 * np.log(probs[0][p2] / occ_prob2 / probs[0][p1] * occ_prob1)
        kl_debias += (tmp_kl1 + tmp_kl2) / 2

        # outputs_P = P.dot(outputs.T).T

        # for a in range(len(A)):
        #     outputs_P = (1 - A[a]) * outputs_P + A[a] * outputs
        #     new_logits = embedding.dot(outputs_P)
        #     new_logits = torch.from_numpy(new_logits).float()
        #     new_logits = new_logits.unsqueeze(0)  # [1, 50257]
        #     probs = F.softmax(new_logits, dim=-1)
        #     probs = probs.cpu().detach().numpy()

    model.lm_head.weight.data = tmp

    return kl, kl_debias
