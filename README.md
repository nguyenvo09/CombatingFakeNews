# Recommending Fact-checking Articles to Combat Fake News
This is the repository for the paper "The Rise of Guardians: Fact-checking URL Recommendation to Combat Fake News" SIGIR 2018,
https://arxiv.org/abs/1806.07516 

## Datasets
- Link to download our full dataset for the paper: You can analyze characteristics of guardians based on this dataset.
https://drive.google.com/file/d/1DQJDMaFJfHo2k6AzEghQXb-xzYbW1Maz/view?usp=sharing 

- Link to download splitted data `Splitted_data.rar`
https://drive.google.com/open?id=1riEsUNP3GHfn7XefuMkH50kW4W2dL0qW

- The splitted data has training, dev and testing interactions. In each part, there are 12,197 guardians with at least
one interaction for each guardian

## Analysis
- Temporal behavior of guardians
![alt text](https://github.com/nguyenvo09/CombatingFakeNews/blob/master/pytorch/images/temporal.png)
- Topical interests of guardians
![alt text](https://github.com/nguyenvo09/CombatingFakeNews/blob/master/pytorch/images/topics.png)

## How to run this code?
- Download the splitted data and extract it. The expected path is `/pytorch/Splitted_data/sigir18/*`
- Then, run the following command with default settings:
```
python Masters/master_gau.py
```
You could achive following performance:
```
|Epoch 11 | Train time: 8 (s) | Train loss: 79212.76166 | Eval time: 30.316 (s) | Vad mapks@10 = 0.06830 | Vad ndcg@10 = 0.08897 | Vad recall@10 = 0.15610 | Test mapks@10 = 0.06879 | Test ndcg@10 = 0.08991 | Test recall@10 = 0.15783
|Epoch 12 | Train time: 8 (s) | Train loss: 75769.19746 | Eval time: 30.028 (s) | Vad mapks@10 = 0.06833 | Vad ndcg@10 = 0.08906 | Vad recall@10 = 0.15635 | Test mapks@10 = 0.06918 | Test ndcg@10 = 0.09030 | Test recall@10 = 0.15832
|Epoch 13 | Train time: 8 (s) | Train loss: 72671.60144 | Eval time: 30.399 (s) | Vad mapks@10 = 0.06876 | Vad ndcg@10 = 0.08946 | Vad recall@10 = 0.15668 | Test mapks@10 = 0.06948 | Test ndcg@10 = 0.09066 | Test recall@10 = 0.15889
|Epoch 14 | Train time: 8 (s) | Train loss: 69873.45222 | Eval time: 29.985 (s) | Vad mapks@10 = 0.06858 | Vad ndcg@10 = 0.08913 | Vad recall@10 = 0.15578 | Test mapks@10 = 0.06952 | Test ndcg@10 = 0.09063 | Test recall@10 = 0.15865
```
## Requirements:
We use PyTorch 0.4.1, Python 3.5. The SPPMI matrices, network and sim matrices are memory-intensive so please run
it on a computer with at least 16GB.


Please cite our paper if you find the data and code helpful, thanks:

```
@inproceedings{vo2018guardians,
	title={The Rise of Guardians: Fact-checking URL Recommendation to Combat Fake News},
	author={Vo, Nguyen and Lee, Kyumin},
	booktitle={The 41st International ACM SIGIR Conference 
		  on Research and Development in Information Retrieval},
	year={2018}
}
```
