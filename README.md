<div align="center">

<samp>
<h2> PitVQA: Image-grounded Text Embedding LLM for Visual Question Answering in Pituitary Surgery </h1>
</samp>     
---
| **[[```arXiv```](<>)]** | **[[```Paper```](<>)]** | **[[```Demo```](<>)]**|
|:-------------------:|:-------------------:|:-------------------:|
    
The International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2024
---

</div> 

## PitVQA Dataset

Our PitVQA dataset comprises 25 videos of endoscopic pi- tuitary surgeries from the The National Hospital of Neurology and Neurosurgery in London, United Kingdom. All patients provided informed consent, and the study was registered with the local governance committee. The surgeries were recorded using a high-definition endoscope (Karl Storz Endoscopy) with a resolu- tion of 720p and stored as MP4 files. All videos were annotated for the surgical phases, steps, instruments present and operation notes guided by a standard- ised annotation framework, which was derived from a preceding international consensus study on pituitary surgery workflow [16]. Annotation was performed collaboratively by 2 neurosurgical residents with operative pituitary experience and checked by an attending neurosurgeon. We extracted image frames from each video at 1 fps and removed any frames that were blurred or occluded. Ul- timately, we obtained a total of 109,173 frames, with the videos of minimum and maximum length yielding 2,443 and 7,179 frames, respectively. We acquired frame-wise question-answer pairs for all the categories of the annotation. Overall, there are 884,242 question-answer pairs from 109,173 frames, which is around 8 pairs for each frame. There are 59 classes overall, including 4 phases, 15 steps, 18 instruments, 3 variations of instruments present in a frame, 5 positions of the instruments, and 14 operation notes in the annotation classes. The length of the questions ranges from a minimum of 7 words to a maximum of 12 words. A comparison of the unique classes between our PitVQA and a publicly avail- able dataset of EndoVis18-VQA is presented in the Table 1. A sample frame and corresponding Q&A pairs are presented in Fig. 1(a). The class frequency distribution is illustrated in Fig. 1(b), where the lowest and highest classes are instruments and phases with 10.04% and 24.70%.

<div align='center'>
<img src='https://github.com/mobarakol/PitVQA/blob/main/assets/pitvqa_dataset_2.png' width=550>
</div>

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
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  pages={},
  year={2024},
  organization={}
}
```
