"""Inference helper for LIS-VLA on LIBERO (same prompt format as RynnVLA-002)."""
import torch
from transformers import GenerationConfig

from libero_util.Chameleon_utils import parse_his_type


def build_prompt_tokens(cur_img, cur_wrist_img, task_description, item_processor, his_img, his_wrist_img,
                        cur_state, his_type):
    d = parse_his_type(his_type)
    if d["his"] == 2 and len(d["views"]) == 3:
        img_c = his_img[-1:] + his_wrist_img[-1:] + [cur_img] + [cur_wrist_img]
    elif d["his"] == 1 and len(d["views"]) == 3:
        img_c = [cur_img] + [cur_wrist_img]
    elif d["his"] == 2 and len(d["views"]) == 2:
        img_c = his_img[-1:] + [cur_img]
    elif d["his"] == 1 and len(d["views"]) == 2:
        img_c = [cur_img]
    else:
        raise ValueError(f"{his_type} not valid!")
    human = f"What action should the robot take to {task_description}?"
    if d["with_state"]:
        human += "<|state|>"
    human += "<|image|>" * len(img_c)
    conv = {"conversations": [{"from": "human", "value": human}], "image": img_c, "action": []}
    if d["with_state"]:
        conv["state"] = cur_state
    tokens = item_processor.process_item(conv, training_mode=False)
    tokens.append(10004)  # action-start token: System 1 / System 2 read the hidden states up to here
    return tokens


@torch.no_grad()
def get_action_lis(model, cur_img, cur_wrist_img, task_description, item_processor, his_img, his_wrist_img,
                   cur_state, his_type, num_candidates=None, cem_iters=None, enable_search=None):
    tokens = build_prompt_tokens(cur_img, cur_wrist_img, task_description, item_processor, his_img,
                                 his_wrist_img, cur_state, his_type)
    gen = GenerationConfig(max_new_tokens=1, max_length=model.config.max_position_embeddings, temperature=1,
                           top_k=None, do_sample=False, eos_token_id=[8710])
    input_ids = torch.tensor(tokens, dtype=torch.int64, device=model.device).unsqueeze(0)
    return model.generate_lis_action(input_ids, gen, num_candidates=num_candidates, cem_iters=cem_iters,
                                     enable_search=enable_search)
