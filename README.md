# Towards Understanding and Mitigating Social Biases in Language Models

This repo contains code and data for evaluating and mitigating bias from generation models.


## Paper

[**Towards Understanding and Mitigating Social Biases in Language Models**](https://arxiv.org/pdf/2106.13219.pdf)<br>
[Paul Pu Liang](http://www.cs.cmu.edu/~pliang/), Chiyu Wu, [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/), and [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/)<br>
ICML 2021

If you find this repository useful, please cite our paper:
```
@article{liang2021towards,
  title={Towards Understanding and Mitigating Social Biases in Language Models},
  author={Liang, Paul Pu and Wu, Chiyu and Morency, Louis-Philippe and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:2106.13219},
  year={2021}
}
```

### 1. Identify bias-sensitive tokens, obtain bias subspace and create the dataset to train the bias classifier
```python
python data_preprocess.py --embed_source glove --by_pca True --num_components 5 --save_subspace False
```

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

It will take a long time to run the evaluation script on the full data. Here we provide the subset of our evaluation data now. Full data will be uploaded via google drive soon.

#### Global Bias

We use the regard score as the metric for global bias. The evaluation code is from https://github.com/ewsheng/nlg-bias. Thanks for their great work!

```python
git clone https://github.com/ewsheng/nlg-bias.git
cd src/global_bias
python generate_full_sentence.py
```

After full sentences are generated, you need to use the regard classifier to measure the global bias. In our experiment, we use the updated classifier.

To reproduce the result in our paper, we also provide the projection matrix P for the gender bias test in `data/saved_P/P_gender_test_79.npy`
