# Recommending Fact-checking Articles to Combat Fake News
This is the repository for the paper "The Rise of Guardians: Fact-checking URL Recommendation to Combat Fake News" SIGIR 2018,
https://arxiv.org/abs/1806.07516 

- Link to download our full dataset for the paper: 
https://drive.google.com/file/d/1DQJDMaFJfHo2k6AzEghQXb-xzYbW1Maz/view?usp=sharing 

- Link to download splitted data `Splitted_data.rar`
https://drive.google.com/open?id=1riEsUNP3GHfn7XefuMkH50kW4W2dL0qW

## How to run this code?
- Download the splitted data and extract it. The expected path is `/pytorch/Splitted_data/sigir18/*`
- Then, run the following command with default settings:
```
python Masters/master_gau.py
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
