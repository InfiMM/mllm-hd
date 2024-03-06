# infimm-hd
Official code for infimm-hd
# Pretrain model
Just download the model at https://huggingface.co/lllliuhhhhggg/infimm_pretrain/tree/main. We provide two pretraining models in our paper (only stage1) https://arxiv.org/abs/2403.01487. It is a Flamingo style model, the only difference is we remove the perceiver resampler. We use vit-e and vicuna in our model.
These models are pretrained on mmc4, obelisc, coyo238m (sampled from coyo700m), laion115m, laioncoco.
# Training
As we utilize the company's training framework, we can not provide the training code directly. So here we give you a demo of the data process and forward pass in demo_forward.py
