# Golden-Gemini-for-Speaker-Verification
🔥🔥🔥 Official release of pretrained models and scripts for ♊ 'Golden Gemini Is All You Need: Finding the Sweet Spots for Speaker Verification' accepted by IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), 2024.

(Free access) IEEE link: https://ieeexplore.ieee.org/document/10497864

arXiv Link: https://arxiv.org/abs/2312.03620

# Note:

1. ***[Important]*** This repository is dedicated to sharing the pretrained models in our paper for connvinent usage. For training and inference, we recommend using the Gemini DF-ResNet, now available on WeSpeaker: https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2. We extend our gratitude to the WeSpeaker community for their support, with special thanks to Dr. Wang Shuai.

2. Special thanks to Dr. Liu Bei for sharing the implementation details related to DF-ResNet (https://ieeexplore.ieee.org/document/10119228).
 
3. [New] We also release a large-margin finetuned pretrained model.

# Pretrained Models

|  Model               | Param | Large Margin Fine-Tuning | Vox1-O EER | Vox1-O MinDCF | Vox1-E EER | Vox1-E MinDCF | Vox1-H EER | Vox1-H MinDCF | Pretained Model Folder                                                                                |
|----------------------|-------|--------------------------|------------|---------------|------------|---------------|------------|---------------|-------------------------------------------------------------------------------------------------------|
|  Gemini DF-ResNet60 [[Google Drive]](https://drive.google.com/file/d/1zfck1eEOFCxGonRRxUzsLKeruwv4f-kU/view?usp=sharing)  | 4.05  |  X                       | 0.941      | 0.089         | 1.051      | 0.116         | 1.799      | 0.166         | 0611-Gemini_df_resnet56-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-AdamW-epoch165     |
|  Gemini DF-ResNet114 [[Google Drive]](https://drive.google.com/file/d/1hruxkctjIzzUkooXikExb3if8wurR6pv/view?usp=sharing) | 6.53  |  X                       | 0.686      | 0.067         | 0.863      | 0.097         | 1.490      | 0.144         | 0615-Gemini_df_resnet110-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-AdamW-epoch165    |
|  Gemini DF-ResNet183 [[Google Drive]](https://drive.google.com/file/d/1Bb1VaD8ZoUREoRoQ73oiCXjIJ21SuKLS/view?usp=drive_link) | 9.20  |  X                       | 0.596      | 0.065         | 0.806      | 0.090         | 1.440      | 0.137         | 0621-Gemini_df_resnet179-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-AdamW-epoch165    |
|  [New] Gemini DF-ResNet183 [[Google Drive]](https://drive.google.com/file/d/1rEb5UpeOvirCt9mhIW54BRAd-6EF3n_c/view?usp=drive_link) | 9.20  | ✔                        | 0.569      | 0.045         | 0.768      | 0.078         | 1.342      | 0.126         | 0621-Gemini_df_resnet179-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-AdamW-epoch165-LM |

*: The layer count between the names of the pretrained model folder and the model itself differs by 4, as mentioned in footnote 4 of the paper, where the distinction lies in whether to include the 4 separate downsampling layers in the layer count.  **The models are identical, only differing in nomenclature.** During experimentation, we did not include the separate downsampling layers in the layer count; however, through discussion during paper writing, we decided to include. **Therefore, Gemini DF-ResNet60/114/183 are the official name.**

# Folder Structure:

Take 0611-Gemini_df_resnet56-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-AdamW-epoch165 as an example:

```
├── models/
│ ├── model_165.pt                                            # The model checkpoint of 165 epoch. This is the model for testing.
├── scores/
│ ├── vox1_asnorm300_result                                   # testing results with asnorm
│ ├── vox1_snorm300_result                                    # testing results with snorm
│ ├── vox1_cos_result                                         # testing results simply by cosine similarity 
│ ├── vox2_dev_asnorm300_vox1_O_cleaned.kaldi.det.png         # visualization
│ ├── vox2_dev_asnorm300_vox1_H_cleaned.kaldi.det.png
│ ├── vox2_dev_asnorm300_vox1_E_cleaned.kaldi.det.png
│ ├── vox2_dev_asnorm300_vox1_O_cleaned.kaldi.score           # scores of all the trials
│ ├── vox2_dev_asnorm300_vox1_H_cleaned.kaldi.score
│ └── vox2_dev_asnorm300_vox1_E_cleaned.kaldi.score
└── config.yaml                                               # The config file to train the model in wespeaker platform (https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2/conf)
└── train.log                                                 # The training log automatically generated by the Wespeaker toolkit. 
```



# Usage:

**[Important]**
🔥🔥🔥 **The Gemini DF-ResNet is now available in Wespeaker! (https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2). We encourage using the version provided by Wespeaker for better compatibility.**
**Additionally, you can find the large-margin finetuned pretrained models in both Pytorch and ONNX formats at: https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md**

Alternatively, you can follow the steps below to reproduce the provided checkpoint:
  1. Set up Wespeaker Toolkit (https://github.com/wenet-e2e/wespeaker).
  2. Copy the model file, Gemini_df_resnet.py, from this repository to wespeaker/wespeaker/models/
  3. Modify wespeaker/wespeaker/models/speaker_model.py by adding
       ```
       import wespeaker.models.Gemini_df_resnet as Gemini_df_resnet
       ```
       and
       ```
       elif model_name.startswith("Gemini_df_resnet"):
         return getattr(Gemini_df_resnet, model_name)
       ```
  4. Create a config file following the config.yaml file in the pre-trained model folder and place it in /wespeaker/examples/voxceleb/v2/conf/
     [Note]: Warm-up is not explicitly stated in the configuration file, yet it is employed by default in the Wespeaker toolkit as follows:
     ```
       warm_from_zero: False
       warm_up_epoch: 6
     ```
  6. In wespeaker/examples/voxceleb/v2/run.sh, modify 'config' to point to the new config file.
  And then you can start training and reproduce.

# Cite
🔥♊**Golden Gemini (this work):**
```  
@ARTICLE{10497864,
  author={Liu, Tianchi and Lee, Kong Aik and Wang, Qiongqiong and Li, Haizhou},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Golden Gemini is All You Need: Finding the Sweet Spots for Speaker Verification}, 
  year={2024},
  volume={32},
  number={},
  pages={2324-2337}
}
```
(Prior Work) RecXi (Golden Gemini is a continued research of tResNet)
```
@inproceedings{NEURIPS2023_9d276b0a,
 author = {Liu, Tianchi and Lee, Kong Aik and Wang, Qiongqiong and Li, Haizhou},
 booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
 pages = {50221--50236},
 title = {Disentangling Voice and Content with Self-Supervision for Speaker Recognition},
 volume = {36},
 year = {2023}
}
```  
(Related Work) DF-ResNet
```  
@ARTICLE{10119228,
  author={Liu, Bei and Chen, Zhengyang and Qian, Yanmin},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Depth-First Neural Architecture With Attentive Feature Fusion for Efficient Speaker Verification}, 
  year={2023},
  volume={31},
  pages={1825-1838}
}
```

(Related Work) wespeaker toolkit
```
@INPROCEEDINGS{10096626,
  author={Wang, Hongji and Liang, Chengdong and Wang, Shuai and Chen, Zhengyang and Zhang, Binbin and Xiang, Xu and Deng, Yanlei and Qian, Yanmin},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Wespeaker: A Research and Production Oriented Speaker Embedding Learning Toolkit}, 
  year={2023},
  pages={1-5}
}
@article{wang4748855advancing,
  title={Advancing Speaker Embedding Learning: Wespeaker Toolkit for Research and Production},
  author={Wang, Shuai and Chen, Zhengyang and Han, Bing and Wang, Hongji and Liang, Chengdong and Zhang, Binbin and Xiang, Xu and Ding, Wen and Rohdin, Johan and Silnova, Anna and others},
  journal={Available at SSRN 4748855}
}
```

