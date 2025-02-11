## Info
**This code is for the paper "Optimistic Interior Point Methods for Sequential Hypothesis Testing by Betting"**, where we refer to some code of [Online Detecting LLM-Generated Texts via Sequential Hypothesis Testing by Betting](https://github.com/canchen-cc/online-llm-detection) and [Sequential Predictive Two-Sample and Independence Testing](https://openreview.net/forum?id=bN1ZBSOV2f).

“Testing by betting" algorithm equipped with our OAlg methods: FTRL+Barrier and Optimistic-FTRL+Barrier, allow updates across the entire interior of the decision space to accelerate the rejection of the null hypothesis without the risk of gradient explosion.

## Synthetic_Datasets
We implemented experiments on three synthetic datasets. 
* Distributions with disjoint supports.
* Distributions with overlapping supports; high signal-to-noise ratio.
* Time-varying distributions with mean shift.


## Detect_LLM
We implemented experiments on existing datasets that consists of scores for texts with the goal of quickly detecting LLM-generated texts.


## Evaluate_Classifier
We implemented experiments for evaluating facial expression classifiers with the goal of training underlying models with fewer samples.


### Citation
If you find this work useful, you can cite it with this BibTex entry:

    @misc{chen2024onlinedetectingllmgeneratedtexts,
          title={Online Detecting LLM-Generated Texts via Sequential Hypothesis Testing by Betting}, 
          author={Can Chen and Jun-Kun Wang},
          year={2024},
          eprint={2410.22318},
          archivePrefix={arXiv},
          primaryClass={cs.LG},
          url={https://arxiv.org/abs/2410.22318}, 
    }
