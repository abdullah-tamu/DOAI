# DOAI (Discrepancy Optimization-Driven Automatic Inpainting)
Facial Anomaly Appraisal Using Discrepancy Optimization-Driven Automatic Inpainting by Abdullah Hayajneh, Erchin Serpedin and Mitchell A. Stotland.

# Requirements
pip install -r requirements.txt

# Usage:
Download the pretrained models and place them in the 'pretrained' directory:
[https://drive.google.com/drive/folders/1tdz1DZa8prWsFLtCc-7bmvyzNAYqppZS?usp=sharing](https://drive.google.com/drive/folders/1tdz1DZa8prWsFLtCc-7bmvyzNAYqppZS?usp=sharing)


An example dataset is found in ./datasets/faces_child

For the main framework with default settings (example dataset), please run:
```
python test_DOAI.py
```
The output heatmaps and the normalized images can be found in the 'result' directory
```
If you find this implementation helpful in your research, please also consider citing:
```
@article{hayajneh2023unsupervised,
  title={Unsupervised anomaly appraisal of cleft faces using a StyleGAN2-based model adaptation technique},
  author={Hayajneh, Abdullah and Shaqfeh, Mohammad and Serpedin, Erchin and Stotland, Mitchell A},
  journal={Plos one},
  volume={18},
  number={8},
  pages={e0288228},
  year={2023},
  publisher={Public Library of Science San Francisco, CA USA}
}
