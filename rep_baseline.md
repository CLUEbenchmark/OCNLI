# Replicate baselines

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

