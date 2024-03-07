# Notice
In this repo, we provide the pretrain code and model. Is utilize resolution of 224. If you want to use the high-resolution model (after all of the four stages), please refer to our hugging face web.
# Pretrain model
Just download the model at https://huggingface.co/lllliuhhhhggg/infimm_pretrain/tree/main. We provide two pretraining models in our paper (only stage1) https://arxiv.org/abs/2403.01487. It is a Flamingo style model, the only difference is we remove the perceiver resampler. We use vit-e and vicuna in our model.
These models are pretrained on mmc4, obelisc, coyo238m (sampled from coyo700m), laion115m, laioncoco. Our model's training speed is much faster than LLaVA due to the cross attention information fusion (using same amount of data). Feel free to build something from our pretrained model. 

# Training
As we utilize the company's training framework, we can not provide the training code directly. So here we give you a demo of the data process and forward pass in demo_forward.py

# License

<a href="https://creativecommons.org/licenses/by-nc/4.0/deed.en">
	<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Cc_by-nc_icon.svg/600px-Cc_by-nc_icon.svg.png" width="160">
</a>
This project is licensed under the **CC BY-NC 4.0**.

The copyright of the images belongs to the original authors.

See [LICENSE](LICENSE) for more information.
# Reference
https://github.com/baaivision/EVA
https://github.com/mlfoundations/open_flamingo
# Citation

```latex
@misc{liu2024infimmhd,
      title={InfiMM-HD: A Leap Forward in High-Resolution Multimodal Understanding}, 
      author={Haogeng Liu and Quanzeng You and Xiaotian Han and Yiqi Wang and Bohan Zhai and Yongfei Liu and Yunzhe Tao and Huaibo Huang and Ran He and Hongxia Yang},
      year={2024},
      eprint={2403.01487},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
