# PitVQA
## PitVQA Dataset
The dataset will be released upon acceptance of the paper
<div align='center'>
<img src='https://github.com/mobarakol/PitVQA/blob/main/assets/Dataset_Annaotation_Classes.png' width=550>
</div>

## Training Command:
For EndoVis18-VQA dataset:
```
python main_endo.py
```

For PitVQA dataset:
```
python main_pit.py
```
## Acknowledgement
The implementation of PitVQA relies on resources from <a href="https://github.com/salesforce/BLIP">BLIP</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a> and our previous work [SurgicalGPT](https://github.com/lalithjets/SurgicalGPT). We thank the original authors for their open-sourcing.

## Citation
If you use this code for your research, please cite our paper.

```
@inproceedings{he2024pitvqa,
  title={PitVQA: Image-grounded Text Embedding LLM for Visual Question Answering in Pituitary Surgery},
  author={He, Runlong and Xu, Mengya and Das, Adrito and Z. Khan, Danyal and Bano, Sophia and J. Marcus, Hani and Stoyanov, Danail and J. Clarkson, Matthew and Islam, Mobarakol},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={},
  year={2024},
  organization={}
}
```
