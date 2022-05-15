import lmdb
import torch
import numpy as np
from lz4.frame import compress, decompress
from utils.template import WTemplate
import msgpack
import msgpack_numpy
from pytorch_pretrained_bert import BertTokenizer
from model.vqa import UniterForVisualQuestionAnswering

with open("vocab.txt", "r") as f:
    vocab = f.read().splitlines()

template = WTemplate()
tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)


def get_input_data():
    txt_env = lmdb.open("/txt/vqa_test.db", readonly=True, create=False)
    txt_txn = txt_env.begin()
    # for key, value in txt_txn.cursor():
    txt_key = b'409703002'
    value = txt_txn.get(txt_key)
    txt_dump = msgpack.loads(decompress(value), raw=False)
    question = txt_dump['question']
    img_key = txt_dump["img_fname"]
    
    img_env = lmdb.open("/img/coco_test2015/feat_th0.2_max100_min10",
                        readonly=True, create=False)
    img_txn = img_env.begin()
    value = img_txn.get(img_key.encode())
    img_dump = msgpack.loads(value, raw=False)

    txt_env.close()
    img_env.close()

    prompt = template.pre_question(question)
    prompt_tokens = tokenizer.tokenize(prompt)
    prompt_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)

    input_ids = torch.tensor([101] + txt_dump['input_ids'] + prompt_ids+ [102]).unsqueeze(0).cuda()
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).cuda()

    num_bb = img_dump['norm_bb'][b'shape'][0]
    img_feat = torch.tensor(
        np.frombuffer(img_dump['features'][b'data'], dtype=np.float16).reshape(num_bb, -1)).unsqueeze(0).cuda().float()
    norm_bb = np.frombuffer(img_dump['norm_bb'][b'data'], dtype=np.float16).reshape(num_bb, -1)
    areas = np.ones(num_bb)
    for i in range(num_bb):
        areas[i] = norm_bb[i][-1] * norm_bb[i][-2]
    norm_bb = np.c_[norm_bb, areas]
    img_pos_feat = torch.tensor(norm_bb).unsqueeze(0).cuda().float()

    attention_mask = torch.ones((1, input_ids.size(1) + num_bb), dtype=torch.long).cuda()
    gather_index = torch.arange(0, attention_mask.size(1), dtype=torch.long).unsqueeze(0).cuda()
    
    return {"input_ids": input_ids,
            "position_ids": position_ids,
            "img_feat": img_feat,
            "img_pos_feat": img_pos_feat,
            "attn_masks": attention_mask,
            "gather_index": gather_index,
            "template_ids":[1]
            }


root = "vqa_output_4tokens-mask"
checkpoint = torch.load(f"{root}/ckpt/model_step_20000.pt")
model = UniterForVisualQuestionAnswering.from_pretrained(f"{root}/log/model.json", checkpoint,img_dim=2048,num_answer=3129)
model.cuda()

batch = get_input_data()

score, attention_prob = model(batch, compute_loss=False)
attention_prob = attention_prob.squeeze().cpu().detach().numpy()
np.save("figures/soft_attention", attention_prob)
# input_ids = batch["input_ids"][-1].cpu().detach().numpy()

# for i, id in enumerate(input_ids):
#     print(f"{i} {vocab[id]}")
