##Model

This is an implementation of Facebook's baseline GRU/LSTM model on the bAbI dataset 
[Weston et al. 2015](https://research.facebook.com/researchers/1543934539189348).

The bAbI dataset contains 20 different question answering tasks.

### Model script

The scripts for this model are included in the neon repo examples directory
([link](https://github.com/NervanaSystems/neon/tree/master/examples/babi)).
See the [readme](https://github.com/NervanaSystems/neon/blob/master/examples/babi/README.md)
there for instructions on how to train the model and run the demo.

### Trained weights
The trained weights file for a GRU network trained on task 15 can be downloaded from AWS
using the following link: [trained model weights on task 15][S3_WEIGHTS_FILE].
[S3_WEIGHTS_FILE]: https://s3-us-west-1.amazonaws.com/nervana-modelzoo/bAbI/babi_task15.p

### neon version
The model weight file above has been generated using neon version tag [v1.4.0]((https://github.com/NervanaSystems/neon/releases/tag/v1.4.0).
It may not work with other versions.

### Performance
Task Number                  | FB LSTM Baseline | Neon QA GRU
---                          | ---              | ---
QA1 - Single Supporting Fact | 50               |  47.9
QA2 - Two Supporting Facts   | 20               |  29.8
QA3 - Three Supporting Facts | 20               |  20.0
QA4 - Two Arg. Relations     | 61               |  69.8
QA5 - Three Arg. Relations   | 70               |  56.4
QA6 - Yes/No Questions       | 48               |  49.1
QA7 - Counting               | 49               |  76.5
QA8 - Lists/Sets             | 45               |  68.9
QA9 - Simple Negation        | 64               |  62.8
QA10 - Indefinite Knowledge  | 44               |  45.3
QA11 - Basic Coreference     | 72               |  67.6
QA12 - Conjunction           | 74               |  63.9
QA13 - Compound Coreference  | 94               |  91.9
QA14 - Time Reasoning        | 27               |  36.8
QA15 - Basic Deduction       | 21               |  52.6
QA16 - Basic Induction       | 23               |  50.1
QA17 - Positional Reasoning  | 51               |  49.0
QA18 - Size Reasoning        | 52               |  90.5
QA19 - Path Finding          | 8                |   9.0
QA20 - Agent's Motivations   | 91               |  95.6

## Citation

[Facebook research](https://research.facebook.com/researchers/1543934539189348)

```
Weston, Jason, et al. "Towards AI-complete question answering: a set of prerequisite toy tasks." arXiv preprint arXiv:1502.05698 (2015).
```
