# Hierarchical and Step-Layer-Wise Tuning of Attention Specialty for Multi-Instance Synthesis in Diffusion Transformers 🎨

[[Project Page]](https://github.com/bitzhangcy/MIS_DiT) [[Paper]](https://arxiv.org/abs/2504.10148) 

## 🔥 News

- 2025-03-25: Our paper with supplementary material [Attention Specialty for Diffusion Transformers](https://arxiv.org/abs/2504.10148) is now available on arXiv.
- 2025-03-25: We release the code!


## 📝 Introduction

A training-free method based on DiT-based models (e.g., FLUX.1.dev, FLUX.1.schnell, SD v3.5) that allows users to precisely place instances and accurately attribute representations in detailed multi-instance layouts using preliminary sketches, while maintaining overall image quality.

## ✅ To-Do List

- [x] Arxiv Paper with Supplementary Material
- [x] Inference Code 
- [ ] More Demos. Coming soon. stay tuned! 🚀
- [ ] ComfyUI support
- [ ] Huggingface Space support

## 🛠️ Installation

### 💻 Environment Setup

```bash
git clone https://github.com/bitzhangcy/MIS_DiT.git
cd MIS_DiT
conda create -n ast python=3.10
conda activate ast
pip install -r requirements.txt
```

### 🚀 Checkpoints

The default checkpoint is **FLUX.1-dev** ([link](https://huggingface.co/black-forest-labs/FLUX.1-dev)). Additionally, [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)  and  [SD v3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) are also supported, with FLUX.1-schnell utilizing different hyperparameters and SD v3.5 featuring a distinct model architecture and parameter set.

Get the access token from [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and set it at line 63 in `flux_hcrt.py` as `hf_token = "your_access_token"`.

## 🎨 Inference

You can quickly perform precise multi-instance synthesis using the following Gradio interface and the instructions below:
```
python flux_hcrt.py
```
#### User Instructions:

- Create the image layout.
<p align="center">
  <img src="./figures/step1.png"style="width:48%"></img>
</p>

- Enter text prompt and label each segment.
<p align="center">
  <img src="./figures/step2.png"style="width:48%"></img>
</p>


- Check the generated images, and tune the hyperparameters if needed.<br>
  w<sup>c</sup> : Degree of T2T attention modulation module. <br>
  w<sup>d</sup> : Degree of I2T attention modulation module. <br>
  w<sup>f</sup> : Degree of I2I attention modulation module. <br>
  
<p align="center">
  <img src="./figures/step4.png" style="width:45%"/></img>
  <img src="./figures/step3.png" style="width:48%"/></img>
</p>


## 📊 Comparison with Other Models

<p align="center">
  <img src="./figures/comparison.png" alt="comparison"/></img>
</p>

## 🤝 Acknowledgement

We sincerely thank the authors of [DenseDiffusion](https://arxiv.org/abs/2308.12964) for their open-source [code](https://github.com/naver-ai/DenseDiffusion), which serves as the foundation of our project. 

## 📚 Citation

If you find this repository useful, please cite using the following BibTeX entry:

```bibtex
@misc{,
      title={Hierarchical and Step-Layer-Wise Tuning of Attention Specialty for Multi-Instance Synthesis in Diffusion Transformers},
      author={Zhang, Chunyang and Sun, Zhenhong and Zhang, Zhicheng and Wang, Junyan and Zhang, Yu and Gong, Dong and Mo, Huadong and Dong, Daoyi},
      year={2025},
      eprint={2504.10148},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.10148},
}
```

## 📬 Contact

If you have any questions or suggestions, please feel free to contact us 😆!