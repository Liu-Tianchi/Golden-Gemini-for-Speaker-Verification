# Golden-Gemini-for-Speaker-Verification
Official release of pretrained models and scripts for 'Golden Gemini Is All You Need: Finding the Sweet Spots for Speaker Verification'

# Note:
1. Special thanks to Dr. Liu Bei for sharing the code related to DF-ResNet (https://ieeexplore.ieee.org/document/10119228).
2. The purpose of this repository is to share pretrained models in the paper. Gemini DF-ResNet is now available on Wespeaker (https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2). Thanks to the Wespeaker community for their assistance.
3. [New] We also release a large-margin finetuned pretrained model.

# Pretrained Models

|  Model               | Param | Large Margin Fine-Tuning | Vox1-O EER | Vox1-O MinDCF | Vox1-E EER | Vox1-E MinDCF | Vox1-H EER | Vox1-H MinDCF | Pretained Model Folder                                                                                |
|----------------------|-------|--------------------------|------------|---------------|------------|---------------|------------|---------------|-------------------------------------------------------------------------------------------------------|
|  Gemini DF-ResNet60  | 4.05  |  X                       | 0.941      | 0.089         | 1.051      | 0.116         | 1.799      | 0.166         | 0611-Gemini_df_resnet56-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-AdamW-epoch165     |
|  Gemini DF-ResNet114 | 6.53  |  X                       | 0.686      | 0.067         | 0.863      | 0.097         | 1.490      | 0.144         | 0615-Gemini_df_resnet110-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-AdamW-epoch165    |
|  Gemini DF-ResNet183 | 9.20  |  X                       | 0.596      | 0.065         | 0.806      | 0.090         | 1.440      | 0.137         | 0621-Gemini_df_resnet179-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-AdamW-epoch165    |
|  Gemini DF-ResNet183 | 9.20  | ✔                        | 0.569      | 0.045         | 0.768      | 0.078         | 1.342      | 0.126         | 0621-Gemini_df_resnet179-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-AdamW-epoch165-LM |







# Usage:

# Cite
**Golden Gemini (this work):**
  @article{liu2023golden,
    title={Golden Gemini is All You Need: Finding the Sweet Spots for Speaker Verification},
    author={Liu, Tianchi and Lee, Kong Aik and Wang, Qiongqiong and Li, Haizhou},
    journal={arXiv preprint arXiv:2312.03620},
    year={2023}
  }

