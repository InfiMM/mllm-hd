import yaml
import torch
from open_clip.transform import image_transform
from PIL import Image
from model import InfiMMModel


### init the model ###
with open('/opt/tiger/infimm-hd/configs/vite_vicuna7b_pretrain_model.yaml', 'r') as f:
    config = yaml.safe_load(f)
model = InfiMMModel(config)

##### Data loading ####

# images loading and processing
img_processor = image_transform(image_size=224, is_train=False) # do not use data augmentation in training as it may destroy the spatial info
image_list = ["1.jpg"]
imgs = [Image.open(img_path) for img_path in image_list]
imgs = [img_processor(img) for img in imgs]
images = torch.stack(imgs) # if the sample is only text, just creat a fake img, e.g. images = torch.randn(1, 3, 224, 224) 
# remember to make the image to [bs, t, f, c, h, w], in llava665k ,t=f=1
images = images.unsqueeze(0) #[bs,t,c,h,w]
images = images.unsqueeze(2) #[bs,t,f,c,h,w]
# <|endofchunk|> replace the eos in flamingo, remember to add the system prompt for Vicuna ""
system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
# assume we have two rounds conversation. And there is only one image as llava665k.
# add <image> token to where it should be in the text sequence, we have mask to process it
qas = [{"question": "<image>How many cats are there?", "answer":"2"}, {"question": "What is the color of the background?", "answer":"pink"}]
tokenizer = model.tokenizer

processed_qas = []

question_lengths = []
round_lengths = []
for round_num, round in enumerate(qas):
    if round_num == 0:
        question = system + ' USER: ' + round['question'] + ' ASSISTANT: '
    else:
        question = 'USER: ' + round['question'] + " " + 'ASSISTANT: '
    question_token = tokenizer(
        [question],
        return_tensors="pt",
    )
    question_length = question_token["input_ids"].shape[1] - 2
    question_lengths.append(question_length)
    # Flamingo takes <eoc> as eos
    answer = round["answer"]
    answer = answer.strip()
    answer_noeos = answer[:]
    round_token = tokenizer(
        [question+answer_noeos],
        return_tensors="pt",
    )
    round_length = round_token["input_ids"].shape[1]
    round_lengths.append(round_length)
    # Add eoc to each round's end 
    answer = answer + '<|endofchunk|>'
    text = question + answer
    processed_qas.append(text)

context_length = 512
text = " ".join(processed_qas)
text_token = tokenizer(
    [text],
    max_length=context_length, # set this yourself
    truncation=True,
    padding="max_length",
    return_tensors="pt",
)
total_length = int(text_token.input_ids.ne(tokenizer.pad_token_id).sum())
question_mask = torch.zeros_like(text_token["attention_mask"][0])
cur_len = 1
question_mask[:cur_len] = 1
for q_len, r_len in zip(question_lengths,  round_lengths):
    question_mask[cur_len:cur_len+q_len] = 1
    cur_len += r_len
# if cur_len < 512:
#     if cur_len != context_length:
#        skip (just fix your dataset)
samples = {}
samples["input_ids"] = text_token["input_ids"]
samples["attention_mask"] = text_token["attention_mask"]
samples["image"] = images
samples["question_mask"] = question_mask.unsqueeze(0)

loss = model(samples)
print(loss)
