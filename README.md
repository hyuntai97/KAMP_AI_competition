# Codebase for our project 
> ## Project title (Kor): 다양한 시나리오로 확장 가능한 준지도 학습 기반  용해공정 불량 탐지 알고리즘 개발
> ## Project title (Eng): Development of a Semi-Supervised Learning-Based Anomaly Detection Algorithm for Raw Material Processing in Melting Tanks, Scalable Across Various Scenarios.

Contact: soodaman97@cau.ac.kr 

### We won the **grand prize** at the 2nd K-Artificial Intelligence Manufacturing Data Analysis Competition!

- Hosted by: 중소벤처기업부(Ministry of SMEs and Startups) | 스마트제조혁신추진단(Smart Manufacturing Innovation Promotion Team) | KAIST

- Competition Link: https://www.kamp-ai.kr/main

- Article Link: [인공지능신문](https://www.aitimes.kr/news/articleView.html?idxno=26717), [한국경제](https://www.hankyung.com/article/202212068219Y), [연합뉴스](https://www.yna.co.kr/view/AKR20221206118000003), [뉴시스](https://mobile.newsis.com/view.html?ar_id=NISX20221206_0002113449), [뉴스1](https://www.news1.kr/photos/details/?5725471), [ITBizNews](https://www.itbiznews.com/news/articleView.html?idxno=84606)

## Report & PPT slide
[![googledrive](https://img.shields.io/badge/report-Link-blue)](https://drive.google.com/file/d/1w4SG_naSJWj9bt_5yokr1_OEmg2F0nFb/view?usp=drive_link)
<br>  

[![googledrive](https://img.shields.io/badge/ppt-Link-blue)](https://drive.google.com/file/d/14o0G3dTUR4kE_2bhbnsUjVscPH5IV2xH/view?usp=drive_link)
<br>  

## Dataset 
- Link: https://www.kamp-ai.kr/aidataDetail?AI_SEARCH=%EC%9A%A9%ED%95%B4%ED%83%B1%ED%81%AC&page=1&DATASET_SEQ=8&EQUIP_SEL=&GUBUN_SEL=&FILE_TYPE_SEL=&WDATE_SEL=

## Time series Anomaly Detection (DeepSAD)
This directory contains implementations of anomaly detection framework for real-world time-series data. Our framework is based on the deep semi-supervised learning algorithm, [DeepSAD](https://arxiv.org/abs/1906.02694).

### Code Explanation

#### Train / Evaluation --argument {default}
```
cd DeepSAD

python main.py --net_name {LSTM} --data{custom} --root_path {./} --data_path {data.csv} --timeenc {1} --seq_len {50} --n_features {4} --embedding_dim {32} --eta {1.0} --pretrain {True} --ae_n_epochs {10} --train_epochs {10} --gpu {5}
```

#### Key Arguments
`--root_path`: str, Data storage directory  

`--xp_path`: str, Directory containing experimental results and training history

`--data_path`: str, Full name of data file  

`--seq_len`: int, Input sequence length   

`--ae_n_epochs`: int, Number of epochs to train autoencoder   

`--train_epochs`: int, Bynber if eoicgs 

`--lr`:  float, learning rate

`--n_features`: int, Number of features in multivariate time series  

`--embedding_dim`: int, Hidden dimensions    

`--n_layers`: int, Number of neural network layers    

`--gpu`: str, Computation device to use, e.g., 5 -> cuda:5

## Time series Forecasting (Autoformer)
This directory contains implementations of time series forecasting framework. Our framework is based on the Transformer, [Autoformer](https://arxiv.org/abs/2106.13008).


## File Directory 

```bash
.
├─── DeepSAD
│    └──  base
│       ├── base_net.py
│       └──  base_trainer.py
│    └──  data_provider
│       ├── data_factory.py
│       └── data_loader.py
│    └──  networks
│       ├── lstm.py
│       └── main.py
│    └──  optim
│       ├── ae_trainer.py
│       └── DeepSAD_trainer.py
│    └──  main.py
│    └──  DeepSAD.py
│
├─── Autoformer 
│    └──  data_provider
│       ├── data_factory.py
│       └── data_loader.py
│    └──  exp
│       ├── exp_basic.py
│       └── exp_main.py
│    └──  layers
│       ├── AutoCorrelation.py
│       ├── Autoformer_EncDec.py
│       ├── Embed.py
│       ├── SelfAttention_Family.py
│       └── Transformer_EncDec.py
│    └──  models
│       ├── Autoformer.py
│       ├── Informer.py
│       ├── Reformer.py
│       └── Transformer.py
│    └──  utils
│       ├── masking.py
│       ├── metrics.py
│       ├── timefeatures.py
│       └── tools.py
│    └──  run.py
```