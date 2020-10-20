# OCNLI: Original Chinese Natural Language Inference

OCNLI stands for **Original** Chinese Natural Language Inference.
It is corpus for Chinese Natural Language Inference, collected following closely the procedures of [MNLI](https://cims.nyu.edu/~sbowman/multinli/), but with enhanced strategies aiming for more challenging inference pairs. 
We want to emphasize we did not use human/machine translation in creating the dataset, and thus our
Chinese texts are *original* and not translated. 

OCNLI has roughly 50k pairs for training, 3k for development and 3k for test. We only release the test data
but not its labels. See our [paper](https://arxiv.org/abs/2010.05444) for details. 

OCNLI is part of the [CLUE](https://www.cluebenchmarks.com/) benchmark.

OCNLI，即原生中文自然语言推理数据集，是第一个非翻译的、使用原生汉语的大型中文自然语言推理数据集。
OCNLI包含5万余训练数据，3千验证数据及3千测试数据。除测试数据外，我们将提供数据及标签。测试数据仅提供数据。OCNLI为中文语言理解基准测评（CLUE）的一部分。

## Data format

Our dataset is distributed in json format. Here's an example from OCNLI.dev:

```
{
"level":"medium",
"sentence1":"身上裹一件工厂发的棉大衣,手插在袖筒里",
"sentence2":"身上至少一件衣服",
"label":"entailment","label0":"entailment","label1":"entailment","label2":"entailment","label3":"entailment","label4":"entailment",
"genre":"lit","prem_id":"lit_635","id":0
}
```
where:

- level: `easy`, `medium` and `hard` refer to the 1st, 2nd and 3rd sentence the annotator wrote respectively for that premise and label. See our paper for details.
- sentence1: the premise sentence(s)
- sentence2: the hypothesis sentence(s)
- label: majority vote from label0 -- label4. If no majority agreement, the label will be `-`, and this example should be excluded in experiments, same as in SNLI and MNLI (already taken care of in our baseline code)
- label0 -- label4: 5 annotated labels for the NLI pair. All pairs in dev and test have 5 labels, whereas only a small portion in the training set has 5 labels
- genre: one of `gov`, `news`, `lit`, `tv` and `phone`
- prem_id: id for the premise
- id: overal id

You will only need `sentence1`, `sentence2` and `label` to train and evaluate. 

## Data split

We provide four training sets: 

1. OCNLI.train: 50k data points
2. OCNLI.train.small: filtered subset of OCNLI.train with 30k data points. 
3. OCNLI.train.10k: 10k data points sampled from OCNLI.train.small
4. OCNLI.train.3k: 3k data points sampled from OCNLI.train.small. 

We wanted to see the effect of training size and overlapping premises on the results. The results trained with the first two training sets are reported in our [paper](https://arxiv.org/abs/2010.05444), along with the details about the splits. The last two sets are intended to mimic situations where annotated data are limited. 

All training sets should be validated on the same dev and test sets.

## Leaderboard 排行榜

OCNLI is part of the CLUE benchmark, which will hold a leaderboard [here](https://www.cluebenchmarks.com/nli.html).
You can submit your results on the test set there. 

注：提交格式：提交一个zip压缩包。里面需要包含如下文件： OCNLI_50k.json, OCNLI_30k.json

## Baselines Models (quick start)

All the code for reproducing the  baseline models is included in `ocnli/`. To run
these experiments, we suggest creating a
[conda environment](https://docs.conda.io/en/latest/miniconda.html),
and setting it up by doing the following:
```
conda create -n ocnli python=3.6.7
conda activate ocnli
pip install -r requirements.txt
```

Alternatively Docker can be used  (see `Dockerfile` to
reproduce our environment; note this uses Python3.5 given its reliance
on `tensorflow/tensorflow:1.12.0-gpu-py3`. Most of our experiments
were run using Docker through the [**Beaker**](https://github.com/allenai/beaker) collaboration tool). 

### mnli baselines

We used some of the baselines from the [original MNLI paper](https://www.aclweb.org/anthology/N18-1101/). A
repurposed version of [the original MNLI code](https://github.com/NYU-MLL/multiNLI)
is included in `ocnli/mnli_code`.

Below is how to train a model
```
python -m ocnli.mnli_code.train_snli \
       cbow \                   ## type of model {cbow,bilstm}
       baseline_model \   ## name of model
       --keep_rate "0.9" \
       --alpha "0.0" \
       --emb_train \
       --datapath data \ ## location of data
       --embed_file_name sgns.merge.char \ ## see link below
       --wdir /path/to/output/directory \ ## where to dump results 
       --train_name ocnli/train.json \
       --dev_name ocnli/dev.json \
       --test_name ocnli/test.json \
       --override
```
where `cbow` can be replaced `{bilstm,esim}` to alternate between
different model types.

Chinese word/character embeddings (given above as `sgns.merge.char`), are used in place
of the original GloVE embeddings and are available from
[here](https://github.com/Embedding/Chinese-Word-Vectors); our exact
embeddings are [hosted on google](https://drive.google.com/file/d/14iVFCdH2k3sFC6SMmeV7aOpHuw8wh2CB/view?usp=sharing).

### Transformer baselines

The code for Transformer baselines are adapted from the CLUE [repository](https://github.com/CLUEbenchmark/CLUE/tree/master/baselines/models)

For example, training a BERT or RoBERTa model can be done in the following way:
```
python -m ocnli.{bert,roberta_wwm_large_ext}.run_classifier \
       --task_name=cmnli \
       --do_train=true \
       --do_eval=true \
       --data_dir=/path/to/do \
       --vocab_file=/path/to/model/vocab \
       --bert_config_file=/path/to/model/bert_config.json \
       --init_checkpoint=/path/to/model/bert_model.ckpt \  ## weights,see below
       --max_seq_length=128 \
       --train_batch_size=32 \
       --learning_rate=2e-5 \
       --num_train_epochs=3.0 \
       --output_dir=/path/to/output/directory \
       --keep_checkpoint_max=1 \
       --save_checkpoints_steps=2500
```
Results are reported with the following hyper-parameters: **ROBERTa**:
`lr=2e-5; batch_size=32; # epochs=3.0`, **BERT:** `lr=2e-5,
batch_size=32, #epochs=3.0` (we generally found models to be stable
across different settings). *Note*: Results using this code might vary
slightly from published numbers due to random initialization; see
**Results** section below. 

Additional switches (not in the original CLUE code): `--partial_input`
(to run the hypothesis-only baselines); `--max_input` (for running the learning curve experiments).

Currently used pre-trained weights (see additional information, models and code on the [CLUE GitHub](https://github.com/CLUEbenchmark/CLUE).)
| model | link |
|:--------------------------:|:-----------------------:|
roberta_wwm_large_ext (weights)  | [*original-link*](https://storage.googleapis.com/chineseglue/pretrain_models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.zip) |
bert (weights)                                   |  [*original-link*](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) |

And evaluation can be done via:
```
python -m ocnli.{bert,roberta_wwm_large_ext}.run_classifier \
       --task_name=cmnli \
       --do_arbitrary="yes" \    # switch indicating random file       
       --data_dir=/path/to/specific/eval/jsonl/file \ # exact file to evaluate
       --vocab_file=/path/to/model/vocab \
       --bert_config_file=/path/to/model/config \
       --init_checkpoint=/path/to/model/ \
       --max_seq_length=128 \
       --output_dir=/path/to/output/directory \
       --model_dir /path/to/specific/checkpoint  ## pointer to model+ckpt
```

To ensure that everything is set up correct, below is an example run
on the OCNLI dev set that uses one RoBERTa checkpoint that can be downloaded from [**here**](https://drive.google.com/file/d/1CjNBFOX9WQy4iP3n4HkPykqK3Qu873__/view?usp=sharing)
```
python -m ocnli.roberta_wwm_large_ext.run_classifier \
       --task_name cmnli \
       --do_arbitrary yes \
       --data_dir path/to/ocnli/dev.json \ ## change here to alternative different different files
       --vocab_file /path/to/pretrained/roberta/above/vocab.txt \
       --bert_config_file  /path/to/pretrained/roberta/above/bert_config.json \
       --max_seq_length 128 \
       --output_dir _runs/ex_roberta_run \
       --model_dir /path/to/checkpoint/above/model.ckpt-4728 \
       --eval_batch_size 1
```
This will generate a file `metrics.json` that should look as follows (where `evaluation_accuracy` is the resulting score):
```
{
    "evaluation_accuracy": 0.7942373156547546,
    "evaluation_loss": 0.7230983376502991
}
```

### Results

- trained with **OCNLI.train = 50k data points**

Accuracy on *dev* / *test* sets: mean accuracy across 5 runs (standard
deviation).

(We will not provide labels for 
the test set. However, you can submit your results on [CLUE](https://www.cluebenchmarks.com/) to obtain test accuracy. **TODO**)

| validation data | majority | CBOW | BiLSTM | ESIM | BERT | RoBERTa | human | 
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| dev  |37.4| 56.8 (0.4) | 60.5 (0.4) | 61.8 (0.5) | 74.5 (0.3) | 78.8 (1.0) | na |
| test |38.1| 55.7 (0.5) | 59.2 (0.5) | 59.8 (0.4) | 72.2 (0.7) | 78.2 (0.7) | 90.3 |


- trained with **OCNLI.train.small = 30k data points**

| validation data | BiLSTM | BERT | RoBERTa | human | 
|:--:|:--:|:--:|:--:|:--:|
| dev  | 58.7 (0.3) | 72.6 (0.9) | 77.4 (1.0) | na |
| test | 57.0 (0.9) | 70.3 (0.9) | 76.4 (1.2) | 90.3 |

## More details about OCNLI

- OCNLI is collected using an **enhanced SNLI/MNLI procedure** where the biggest difference is that
the annotators were instrcucted to write **3 hypotheses per label per premise**, instead of 1. 
That is, an annotator needs to produce 3 entailed hypotheses, 3 neutral ones and 3 contradictions, 
for a given premise. We believe this forces the annotators to produce more challening hypotheses.
At the time of publication, there is a 13% gap between human performance and that of the best model.
【OCNLI改进了SNLI、MNLI数据收集和标注方法，使数据难度更大，对现有模型更有挑战性。目前(2020年10月)人类测评得分比模型最高分高出12%。】

- Our premises come from **5 genres**: government documents, news, literature, TV show transcripts, and
telephone conversation transcripts. 【OCNLI的前提(premise)选自5种不同的文体：政府公文、新闻、文学、电视谈话节目、电话录音。】

- Similar to SNLI/MNLI, we selected a portion of the collected premise-hypothesis pairs for relabelling as sanity check,
and all our dev and test data have received at least 3 out of 5 majority votes. The 'bad' ones have a label of '-' and should 
be excluded in experiments. Our annotator agreement is slightly better than SNLI/MNLI in 2
out of the 4 annotation conditions and similar to SNLI/MNLI in the other 2. 【与SNLI、MNLI类似，我们选取了部分数据进行二次标注，以确保标签的准确性。所有验证和测试数据的标签均为3/5多数投票决定，不合格的数据点标签为"-"，实验中应将这些数据排除。】

- We believe our dataset is **challenging and of high-quality**. 
This is due in no small part to all of our annotators, who are undergraduate students studying language-related subjects in Chinese universities, rather than crowd workers. 【为了保证数据质量，我们的标注人员均为语言相关专业的本科生。OCNLI的完成离不开所有参与标注同学的辛勤努力，我们在此表示感谢！】

- Example pairs from OCNLI:

|sentence1 | sentence2 | source | label |
|:--:|:--:|:--:|:--:|
| 但是不光是中国,日本,整个东亚文化都有这个特点就是被权力影响很深 | 有超过两个东亚国家有这个特点 | OCNLI | E |
| 完善加工贸易政策体|贸易政策体系还有不足之处 | OCNLI| E |
| 咖啡馆里面对面坐的年轻男女也是上一代的故事,她已是过来人了 | 男人和女人是背对背坐着的 | OCNLI | C |
| 今天,这一受人关注的会议终于在波恩举行 | 这一会议原定于昨天举行 | OCNLI | N|
| 嗯,今天星期六我们这儿,嗯哼. | 昨天是星期天 | OCNLI | C|


## Why not XNLI? 

While XNLI has been helpful in multi-lingual NLI research, the quality of XNLI Chinese data is far from satisfactory; here are just a few bad examples we found when annotating 300 randomly sampled examples from XNLI dev:

|sentence1 | sentence2 |  source | label |
|:--:|:--:|:--:|:--:|
|Louisa May Alcott和Nathaniel Hawthorne 住在Pinckney街道，而 那个被Oliver Wendell Holmes称为 “晴天街道 的Beacon Street街道住着有些喜欢自吹自擂的历史学家 William Prescott	|Hawthorne住在Main Street上	|XNLI dev|	C|
看看东方的Passeig de Gracia，特别是Diputacie，Consell de Cent，Mallorca和Valancia，直到Mercat de la Concepcie市场|	市场出售大量的水果和蔬菜|	XNLI dev|	N|
Leisure Modern medicine and hygiene学说已经解决了过去占据我们免疫系统的大部分问题	|人类是唯一没有免疫系统的生物|	XNLI dev|	C|
政府，法律的batta, begar, chaprasi, dakoit, dakoity, dhan, dharna, kotwal, kotwali, panchayat, pottah, sabha|	所有的单词都很容易理解|	XNLI dev|	C|
下一阶段，中情局基地组织的负责人当时回忆说，他不认为他的职责是指导应该做什么或不应该做什么|	导演认为这完全取决于他|	XNLI dev|	C|

## Related resources

- [CLUE](https://www.cluebenchmarks.com/): Chinese Language Understanding Evaluation benchmark
- [SNLI](https://nlp.stanford.edu/projects/snli/): Stanford NLI corpus
- [MNLI](https://cims.nyu.edu/~sbowman/multinli/): Multi-genre NLI corpus
- [XNLI](https://cims.nyu.edu/~sbowman/xnli/): Cross-Lingual NLI corpus
- [ANLI](https://github.com/facebookresearch/anli): Adversarial NLI corpus

## TODO

- set up submission of test results on CLUE
- code for baseline models in Huggingface [Feel free to make a PR]

## Contributors

Hai Hu, Kyle Richardson, Liang Xu, Lu Li, Sandra Kuebler and Larry Moss

## Acknowledgements

We greatly appreciate the hard work of our annotators, who are from the following universities:
Xiamen University, Beijing Foreign Studies University, University of Electronic Science and Technology of China, and Beijing Normal University. 
We also want to thank Ruoze Huang, Zhaohong Wu, Jueyan Wu and Xiaojie Gong for helping us to find the annotators. 
This project is funded by Grant-in-Aid of Doctoral Research from Indiana University Graduate School and the CLUE benchmark.

## License

- Attribution-NonCommercial 2.0 Generic (CC BY-NC 2.0)
- The premises in the news genre are sampled from the LCMC corpus (ISLRN ID: 990-638-120-277-2, ELRA reference: ELRA-W0039), with permission from ELRA.

## Citation

Please cite the following paper if you use OCNLI in your research
```
@inproceedings{ocnli,
	title={OCNLI: Original Chinese Natural Language Inference},
	author={Hai Hu and Kyle Richardson and Liang Xu and Lu Li and Sandra Kuebler and Larry Moss},
	booktitle={Findings of EMNLP},
	year={2020},
	url={https://arxiv.org/abs/2010.05444}
}
```
