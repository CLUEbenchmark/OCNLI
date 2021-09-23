# OCNLI: Original Chinese Natural Language Inference

OCNLI stands for **Original** Chinese Natural Language Inference.
It is corpus for Chinese Natural Language Inference, collected following closely the procedures of [MNLI](https://cims.nyu.edu/~sbowman/multinli/), but with enhanced strategies aiming for more challenging inference pairs. 
We want to emphasize we did not use human/machine translation in creating the dataset, and thus our
Chinese texts are *original* and not translated. 

OCNLI has roughly 50k pairs for training, 3k for development and 3k for test. We only release the test data
but not its labels. See our [paper](https://arxiv.org/abs/2010.05444) for details. 

OCNLI is part of the [CLUE](https://www.cluebenchmarks.com/) benchmark.

OCNLI，即原生中文自然语言推理数据集，是第一个非翻译的、使用原生汉语的大型中文自然语言推理数据集。
OCNLI包含5万余训练数据，3千验证数据及3千测试数据。除测试数据外，我们将提供数据及标签。测试数据仅提供数据。OCNLI为中文语言理解基准测评（[CLUE](https://www.cluebenchmarks.com/)）的一部分。更多细节请参考我们的[论文](https://arxiv.org/abs/2010.05444)。

## Data format 数据格式

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

- `level`: `easy`, `medium` and `hard` refer to the 1st, 2nd and 3rd hypothesis the annotator wrote respectively for that premise and label. See our paper for details. 【难度】: `easy`, `medium`, `hard`分别代表标注人员为某一标签（如entailment）写的第一、第二、第三个假设。我们预计三者难度递增。具体数据收集方式请参考论文。
- `sentence1`: the premise sentence(s) 【句子1】，即前提。
- `sentence2`: the hypothesis sentence(s)  【句子2】，即假设。
- `label`: majority vote from label0 -- label4. If no majority agreement, the label will be `-`, and this example should be excluded in experiments, same as in SNLI and MNLI (already taken care of in our baseline code) 【标签】，即标签0 -- 标签4的majority vote。如果标签为'-'，则此数据应除去，因为5名标注人员没有得出共识，此项设置与SNLI/MNLI相同。
- `label0` -- `label4`: 5 annotated labels for the NLI pair. All pairs in dev and test have 5 labels, whereas only a small portion in the training set has 5 labels 【5个标签】，验证集与测试集的数据均有5个标签。训练集仅部分数据有5个标签。
- `genre`: one of `gov`, `news`, `lit`, `tv` and `phone`  【文本类别】，共5类：政府公报、新闻、文学、电视谈话节目、电话转写。
- `prem_id`: id for the premise 【前提编号】
- `id`: overall id 【总编号】

You will only need `sentence1`, `sentence2` and `label` to train and evaluate. 训练和验证时仅需【sentence1】【sentence2】【label】

## Data split 数据集切分

为了了解训练集大小对结果的影响，我们提供四个大小不同的训练集。`OCNLI.train.3k`, `OCNLI.train.10k`, `OCNLI.train.30k`均为 `OCNLI.train.50k`的子集。四种情况下的验证和测试集相同，均为`OCNLI.dev`, `OCNLI.test`. 

We provide four training sets: 

1. `OCNLI.train.50k`: 50k data points (`OCNLI.train` in our paper)
2. `OCNLI.train.30k`: filtered subset of `OCNLI.train.50k` with 30k data points (`OCNLI.train.small` in our paper)
3. `OCNLI.train.10k`: 10k data points sampled from `OCNLI.train.30k`
4. `OCNLI.train.3k`: 3k data points sampled from `OCNLI.train.30k`

We wanted to see the effect of training size and overlapping premises on the results. The results trained with the first two training sets are reported in our [paper](https://arxiv.org/abs/2010.05444), along with the details about the splits. The last two sets are intended to mimic situations where annotated data are limited. 

All training sets should be validated on the same dev and test sets.

## Leaderboard 排行榜

OCNLI is part of the CLUE benchmark, which will hold a leaderboard [here](https://www.cluebenchmarks.com/nli.html).
You can submit your results on the test set there. 

目前可以提交用`OCNLI.train.50k`，`OCNLI.train.30k`训练后的测试结果。

注：提交格式：提交一个zip压缩包。里面需要包含如下文件： OCNLI_50k.json, OCNLI_30k.json

## Baselines 基线模型及结果

### Models

Please refer to https://github.com/CLUEbenchmark/OCNLI/blob/main/rep_baseline.md

### Results

- trained with **OCNLI.train = 50k data points**

Accuracy on *dev* / *test* sets: mean accuracy across 5 runs (standard
deviation). BERT: BERT_base, RoBERTa: RoBERTa_large_wwm. Check more details on the paper.

| validation data | majority | CBOW | BiLSTM | ESIM | BERT | RoBERTa | human | 
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| dev  |37.4| 56.8 (0.4) | 60.5 (0.4) | 61.8 (0.5) | 74.5 (0.3) | 78.8 (1.0) | na |
| test |38.1| 55.7 (0.5) | 59.2 (0.5) | 59.8 (0.4) | 72.2 (0.7) | 78.2 (0.7) | 90.3 |

- trained with **OCNLI.train.small = 30k data points**

| validation data | BiLSTM | BERT | RoBERTa | human | 
|:--:|:--:|:--:|:--:|:--:|
| dev  | 58.7 (0.3) | 72.6 (0.9) | 77.4 (1.0) | na |
| test | 57.0 (0.9) | 70.3 (0.9) | 76.4 (1.2) | 90.3 |

- trained with **OCNLI.train.small = 10k data points**

| validation data | BERT | RoBERTa | human | 
|:--:|:--:|:--:|:--:|
| dev  | 69.2 (0.5) | 75.2 (0.3) | na |
| test | 67.0 (0.6) | 73.6 (0.5) | 90.3 |

- trained with **OCNLI.train.small = 3k data points**

| validation data | BERT | RoBERTa | human | 
|:--:|:--:|:--:|:--:|
| dev  | 64.4 (0.7) | 70.4 (0.6) | na |
| test | 62.8 (0.7) | 69.5 (0.5) | 90.3 |

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
