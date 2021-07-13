# Towards Understanding and Mitigating Social Biases in Language Models

This repo contains code and data for evaluating and mitigating bias from generation models.


## Paper

[**Towards Understanding and Mitigating Social Biases in Language Models**](https://arxiv.org/pdf/2106.13219.pdf)<br>
[Paul Pu Liang](http://www.cs.cmu.edu/~pliang/), Chiyu Wu, [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/), and [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/)<br>
ICML 2021

If you find this repository useful, please cite our paper:
```
@inproceedings{liang2021towards,
  title={Towards Understanding and Mitigating Social Biases in Language Models},
  author={Liang, Paul Pu and Wu, Chiyu and Morency, Louis-Philippe and Salakhutdinov, Ruslan},
  booktitle={International Conference on Machine Learning},
  pages={6565--6576},
  year={2021},
  organization={PMLR}
}
```

### 1. Identify bias-sensitive tokens, obtain bias subspace and create the dataset to train the bias classifier
```python
python data_preprocess.py --embed_source glove --by_pca True --num_components 5 --save_subspace False
```

Glove embedding and gpt2 embedding are large files, you can download or extract them by yourself. We also provide the [google drive link](https://drive.google.com/drive/folders/1up_8TC3_RxyDcmTrm9GKk1rU3dAt76ND?usp=sharing).

### 2. Train the bias classifier and learn the projection matrix P
```python
python context_nullspace_projection.py
```
The code of nullspace projection is from [INLP](https://github.com/shauli-ravfogel/nullspace_projection). Thanks for their great work!

To run the INLP experiments, you need to git clone https://github.com/shauli-ravfogel/nullspace_projection first, and put it under the root directory of this repo.

### 3. Evaluate Bias existing in the gpt2
#### Local Bias
```python
cd src/local_bias
python measure_local_bias.py
```

It will take long time to run the evaluation script on the full data. Here we provide the subset of our evaluation data now. Full data will be uploaded via google drive soon.

#### Global Bias

We use the regard score difference as the metric for global bias. The evaluation code is from https://github.com/ewsheng/nlg-bias. Thanks for their great work!

```python
git clone https://github.com/ewsheng/nlg-bias.git
cd src/global_bias
python generate_full_sentence.py --algorithm INLP
```

After full sentences are generated, you need to use the regard classifier to measure the global bias. 

To reproduce the result in our paper, we also provide the projection matrix P for the gender bias test in `data/saved_P/P_gender_test_79.npy`

## Acknowledgements
