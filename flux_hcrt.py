import gradio as gr
import torch
import os
import pickle
from datetime import datetime
import pdb

from diffusers import FluxPipeline
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union
from diffusers.models.attention_processor import Attention, apply_rope

from utils import preprocess_mask, process_sketch, process_prompts, process_example, attribute_instance


#################################################
#################################################
canvas_html = "<div id='canvas-root' style='max-width:400px; margin: 0 auto'></div>"
load_js = """
async () => {
const url = "https://huggingface.co/datasets/radames/gradio-components/raw/main/sketch-canvas.js"
fetch(url)
  .then(res => res.text())
  .then(text => {
    const script = document.createElement('script');
    script.type = "module"
    script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
    document.head.appendChild(script);
  });
}
"""

get_js_colors = """
async (canvasData) => {
  const canvasEl = document.getElementById("canvas-root");
  return [canvasEl._data]
}
"""

css = '''
#color-bg{display:flex;justify-content: center;align-items: center;}
.color-bg-item{width: 100%; height: 32px}
#main_button{width:100%}
<style>
'''

MAX_COLORS = 12


with open('./dataset/valset.pkl', 'rb') as f:
    val_prompt = pickle.load(f)
val_layout = './dataset/valset_layout/'


#################################################
#################################################
global t2treg, i2treg, i2ireg
global sizereg_image, sizereg_text, decay_size, COUNT, num_inference_steps, num_mod_steps
global reg_text_sizes, sreg_text_maps, reg_image_sizes, sreg_image_maps
global reg_image_sizes_bakcground, reg_image_sizes_instance, pww_maps_background, pww_maps_attribute, pww_maps_instance

device="cuda:0"
hf_token = ""
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, token = hf_token).to(device)# for FLUX.1-dev
# for FLUX.1-schnell without permission token
# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16.to(device)
t2treg = 0
i2treg = 0
i2ireg = 0
sizereg_image = 0

sizereg_text = 0
decay_size = 0
COUNT = 0
num_inference_steps = 0
num_mod_steps = 0
reg_text_sizes = None
sreg_text_maps = None
reg_image_sizes = None
sreg_image_maps = None
reg_image_sizes_bakcground = None
reg_image_sizes_instance = None
pww_maps_background = None
pww_maps_attribute = None
pww_maps_instance = None


#################################################
#################################################
def mod_forward(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
    torch.set_grad_enabled(False)
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    context_input_ndim = encoder_hidden_states.ndim
    if context_input_ndim == 4:
        batch_size, channel, height, width = encoder_hidden_states.shape
        encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2) # 1 512 3072

    batch_size = encoder_hidden_states.shape[0]

    global t2treg, i2treg, i2ireg
    global sizereg_image, sizereg_text, decay_size, COUNT, num_inference_steps, num_mod_steps
    global reg_text_sizes, sreg_text_maps, reg_image_sizes, sreg_image_maps
    global reg_image_sizes_bakcground, reg_image_sizes_instance, pww_maps_background, pww_maps_attribute, pww_maps_instance
    
    # `sample` projections.
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states) # 1*4096*3072
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads # 128 24

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # `context` projections.
    encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
    encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
    encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states) # 1 512 3072

    encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)
    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)
    encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2) # 1 24 512 128

    if attn.norm_added_q is not None:
        encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
    if attn.norm_added_k is not None:
        encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

    # attention
    query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
    key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
    value = torch.cat([encoder_hidden_states_value_proj, value], dim=2) # torch.Size([1, 24, 4608, 128])
    

    if image_rotary_emb is not None:
        query, key = apply_rope(query, key, image_rotary_emb) # torch.Size([1, 24, 4608, 128])  torch.Size([1, 24, 4608, 128]) 
    
    if  ((COUNT - 1) // 57 + 1) <= num_mod_steps and (5 < ((COUNT - 1) % 57 + 1)):
        dtype = query.dtype
        query = query.view(-1, query.shape[2], query.shape[3]) # torch.Size([24, 4608, 128])
        key = key.view(-1, key.shape[2], key.shape[3])# torch.Size([24, 4608, 128])
        value = value.view( -1, value.shape[2], value.shape[3])  # torch.Size([24, 4608, 128])
        scale_factor = torch.sqrt(torch.tensor(1/query.size(-1), dtype=dtype))
        sim = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], 
                                        dtype=dtype, device=query.device),
                            query, key.transpose(-1, -2), beta=0, alpha=scale_factor) # 24 4608 4608 [24, 512:, 512: ]
        attention_probs = F.softmax(sim, dim= -1) # torch.Size([24, 4608, 4608])
        
        treg = torch.pow(pipe.scheduler.timesteps[(COUNT - 1)//57]/1000, decay_size)
        
        # image to image self attention:
        sim_image_image= attention_probs[:, 512:, 512:] # torch.Size([24, 4096, 4096])

        size_image_reg = reg_image_sizes.repeat(attn.heads,1,1) # torch.Size([24, 4096, 1])
        softmax_image_image = sreg_image_maps.repeat(attn.heads,1,1) # torch.Size([24, 4096, 4096])
        softmax_modified_image_image = size_image_reg * i2ireg * treg * (softmax_image_image - sim_image_image) # torch.Size([24, 4096, 4096])
        exp_softmax_modified_image_image = torch.exp(softmax_modified_image_image) # torch.Size([24, 4096, 4096])
        sim_image_image = exp_softmax_modified_image_image * sim_image_image # torch.Size([24, 4096, 4096])

        attention_probs[:, 512:, 512:] = sim_image_image
        
        # text to text self attention 
        sim_text_text = attention_probs[:, :512, :512]
        
        size_text_reg = reg_text_sizes.repeat(attn.heads,1,1) # torch.Size([24, 4096, 1])        
        softmax_text_text = sreg_text_maps.repeat(attn.heads,1,1) # torch.Size([24, 4096, 4096])
        softmax_modified_text_text = size_text_reg * t2treg * treg * (softmax_text_text - sim_text_text) # torch.Size([24, 4096, 4096])
        exp_softmax_modified_text_text = torch.exp(softmax_modified_text_text) # torch.Size([24, 4096, 4096])
        sim_text_text = exp_softmax_modified_text_text * sim_text_text # torch.Size([24, 4096, 4096])
        
        attention_probs[:, :512, :512] = sim_text_text
        
        # image to text cross attention for instance
        sim_image_text= attention_probs[:, 512:, :512] # torch.Size([24, 4096, 512])
        
        softmax_image_text = pww_maps_instance.repeat(attn.heads,1,1) # torch.Size([24, 4096, 512])
        size_image_text = reg_image_sizes_instance.repeat(attn.heads,1,1) # torch.Size([24, 4096, 4096])
        softmax_modified_image_text = size_image_text * i2treg * treg * (softmax_image_text - sim_image_text) # torch.Size([24, 4096, 4096])
        exp_softmax_modified_image_text = torch.exp(softmax_modified_image_text) # torch.Size([24, 4096, 4096])
        sim_image_text = exp_softmax_modified_image_text * sim_image_text # torch.Size([24, 4096, 4096])

        attention_probs[:, 512:, :512] = sim_image_text

        #attention_probs = F.softmax(attention_probs, dim= -1)
        sum_dim = attention_probs.sum(dim=-1, keepdim=True)
        attention_probs = attention_probs / (sum_dim + 1e-12)
        hidden_states = torch.bmm(attention_probs, value).to(dtype)
        hidden_states= hidden_states.view(batch_size, -1, hidden_states.shape[1], hidden_states.shape[2])

    else:
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        
    COUNT += 1
        
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) # torch.Size([1, 4608, 3072])
    hidden_states = hidden_states.to(query.dtype)
    encoder_hidden_states, hidden_states = (
        hidden_states[:, : encoder_hidden_states.shape[1]],
        hidden_states[:, encoder_hidden_states.shape[1] :],
    )
    

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states) # 1 4096 3072 
    encoder_hidden_states = attn.to_add_out(encoder_hidden_states) # 1 512 3072

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
    if context_input_ndim == 4:
        encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
    
    return hidden_states, encoder_hidden_states

for _module in pipe.transformer.modules():
    if _module.__class__.__name__ == "FluxTransformerBlock":
        _module.attn.processor.__class__.__call__ = mod_forward


def single_mod_forward( 
        self,
        attn: Attention,
        hidden_states: torch.Tensor,# 1 4608 3072 
        encoder_hidden_states: Optional[torch.Tensor] = None, # None
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    torch.set_grad_enabled(False)
    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    
    global t2treg, i2treg, i2ireg
    global sizereg_image, sizereg_text, decay_size, COUNT, num_inference_steps, num_mod_steps
    global reg_text_sizes, sreg_text_maps, reg_image_sizes, sreg_image_maps
    global reg_image_sizes_bakcground, reg_image_sizes_instance, pww_maps_background, pww_maps_attribute, pww_maps_instance
    
    query = attn.to_q(hidden_states)
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # torch.Size([1, 24, 4608, 128]) torch.Size([1, 4608, 3072])

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    if image_rotary_emb is not None:
        query, key = apply_rope(query, key, image_rotary_emb) #torch.Size([1, 24, 4608, 128]) torch.Size([1, 24, 4608, 128])

    if  ((COUNT - 1) // 57 + 1) <= num_mod_steps:
        dtype = query.dtype
        query = query.view(-1, query.shape[2], query.shape[3]) # torch.Size([24, 4608, 128])
        key = key.view(-1, key.shape[2], key.shape[3])# torch.Size([24, 4608, 128])
        value = value.view( -1, value.shape[2], value.shape[3])  # torch.Size([24, 4608, 128])
        scale_factor = torch.sqrt(torch.tensor(1/query.size(-1), dtype=dtype))
        sim = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], 
                                        dtype=dtype, device=query.device),
                            query, key.transpose(-1, -2), beta=0, alpha=scale_factor) # 24 4608 4608 [24, 512:, 512: ]
        attention_probs = F.softmax(sim, dim= -1) # torch.Size([24, 4608, 4608])
        
        treg = torch.pow(pipe.scheduler.timesteps[(COUNT - 1)//57]/1000, decay_size)
        
        # image to image self attention:
        if (COUNT - 1) % 57 + 1  <= 50:
            sim_image_image= attention_probs[:, 512:, 512:] # torch.Size([24, 4096, 4096])
            
            size_image_reg = reg_image_sizes.repeat(attn.heads,1,1) # torch.Size([24, 4096, 1])
            softmax_image_image = sreg_image_maps.repeat(attn.heads,1,1) # torch.Size([24, 4096, 4096])
            softmax_modified_image_image = size_image_reg * i2ireg * treg * (softmax_image_image - sim_image_image) # torch.Size([24, 4096, 4096])
            exp_softmax_modified_image_image = torch.exp(softmax_modified_image_image) # torch.Size([24, 4096, 4096])
            sim_image_image = exp_softmax_modified_image_image * sim_image_image # torch.Size([24, 4096, 4096])

            attention_probs[:, 512:, 512:] = sim_image_image
        
        # text to text self attention
        if (COUNT - 1) % 57 + 1  <= 50:
            sim_text_text = attention_probs[:, :512, :512]
            
            size_text_reg = reg_text_sizes.repeat(attn.heads,1,1) # torch.Size([24, 4096, 1])        
            softmax_text_text = sreg_text_maps.repeat(attn.heads,1,1) # torch.Size([24, 4096, 4096])
            softmax_modified_text_text = size_text_reg * t2treg * treg * (softmax_text_text - sim_text_text) # torch.Size([24, 4096, 4096])
            exp_softmax_modified_text_text = torch.exp(softmax_modified_text_text) # torch.Size([24, 4096, 4096])
            sim_text_text = exp_softmax_modified_text_text * sim_text_text # torch.Size([24, 4096, 4096])
            
            attention_probs[:, :512, :512] = sim_text_text
            
        # image to text cross attention
        if (COUNT - 1) % 57 + 1  <= 24:
            # for instance and background
            sim_image_text= attention_probs[:, 512:, :512] # torch.Size([24, 4096, 512])
        
            pww_map_instance_background = pww_maps_instance + pww_maps_background
            softmax_image_text = pww_map_instance_background.repeat(attn.heads,1,1) # torch.Size([24, 4096, 512])
            size_image_text = reg_image_sizes.repeat(attn.heads,1,1) # torch.Size([24, 4096, 4096])
            softmax_modified_image_text = size_image_text * i2treg * treg * (softmax_image_text - sim_image_text) # torch.Size([24, 4096, 4096])
            exp_softmax_modified_image_text = torch.exp(softmax_modified_image_text) # torch.Size([24, 4096, 4096])
            sim_image_text = exp_softmax_modified_image_text * sim_image_text # torch.Size([24, 4096, 4096])
            
            attention_probs[:, 512:, :512] = sim_image_text
        
        if 25 <= (COUNT - 1) % 57 + 1  <= 34:
            # for attribute and instance
            sim_image_text= attention_probs[:, 512:, :512] # torch.Size([24, 4096, 512])
            
            pww_map_instance_attribute = pww_maps_instance + pww_maps_attribute
            softmax_image_text = pww_map_instance_attribute.repeat(attn.heads,1,1) # torch.Size([24, 4096, 4096])
            size_image_text = reg_image_sizes_instance.repeat(attn.heads,1,1) # torch.Size([24, 4096, 4096])
            softmax_modified_image_text = size_image_text * i2treg * treg * (softmax_image_text - sim_image_text) # torch.Size([24, 4096, 4096])
            exp_softmax_modified_image_text = torch.exp(softmax_modified_image_text) # torch.Size([24, 4096, 4096])
            sim_image_text = exp_softmax_modified_image_text * sim_image_text # torch.Size([24, 4096, 4096])

        
            attention_probs[:, 512:, :512] = sim_image_text

        if 35 <= (COUNT - 1) % 57 + 1:
            # for attribute
            sim_image_text= attention_probs[:, 512:, :512] # torch.Size([24, 4096, 512])
            
            softmax_image_text = pww_maps_attribute.repeat(attn.heads,1,1) # torch.Size([24, 4096, 4096])
            size_image_text = reg_image_sizes_instance.repeat(attn.heads,1,1) # torch.Size([24, 4096, 4096])
            softmax_modified_image_text = size_image_text * i2treg * treg * (softmax_image_text - sim_image_text) # torch.Size([24, 4096, 4096])
            exp_softmax_modified_image_text = torch.exp(softmax_modified_image_text) # torch.Size([24, 4096, 4096])
            sim_image_text = exp_softmax_modified_image_text * sim_image_text # torch.Size([24, 4096, 4096])
        
            attention_probs[:, 512:, :512] = sim_image_text       
                 
        #attention_probs = F.softmax(attention_probs, dim = -1)
        sum_dim = attention_probs.sum(dim=-1, keepdim=True)
        attention_probs = attention_probs / (sum_dim + 1e-12)
        hidden_states = torch.bmm(attention_probs, value).to(dtype)
        hidden_states= hidden_states.view(batch_size, -1, hidden_states.shape[1], hidden_states.shape[2])
        
    else:
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    
    COUNT += 1
    
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
    return hidden_states


for _module in pipe.transformer.modules():
    if _module.__class__.__name__ == "FluxSingleTransformerBlock":
        _module.attn.processor.__class__.__call__ = single_mod_forward


#################################################
#################################################
def process_generation(binary_matrixes, seed, num_inference_steps_, num_mod_steps_, height_, width_, t2treg_, i2treg_,i2ireg_, sizereg_image_, sizereg_text_, decay_size_, master_prompt, *prompts):
    torch.set_grad_enabled(False)
    global t2treg, i2treg, i2ireg
    global sizereg_image, sizereg_text, decay_size, COUNT, num_inference_steps, num_mod_steps
    global reg_text_sizes, sreg_text_maps, reg_image_sizes, sreg_image_maps
    global reg_image_sizes_bakcground, reg_image_sizes_instance, pww_maps_background, pww_maps_attribute, pww_maps_instance
    
    t2treg, i2treg,i2ireg = t2treg_, i2treg_,i2ireg_
    num_inference_steps, sizereg_image, sizereg_text = num_inference_steps_, sizereg_image_, sizereg_text_
    height, width = height_, width_ 
    num_mod_steps = num_mod_steps_
    decay_size = decay_size_
    bsz = 1
    COUNT = 1
    res = (height//pipe.vae_scale_factor, width // pipe.vae_scale_factor)
    clipped_prompts = prompts[:len(binary_matrixes)]
    prompts = [master_prompt] + list(clipped_prompts)
    
    
    text_input = pipe.tokenizer_2(
            prompts,
            padding="max_length",
            max_length= 512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
    prompt_embeds = pipe.text_encoder_2(text_input.input_ids.to(device), output_hidden_states=False)[0]    
    _, pooled_prompt_embedding, _= pipe.encode_prompt(prompts[0], prompt_2 = prompts[0])
    
    
    ###########################
    ###### prep for image to image sreg ###### 
    ###########################
    layouts = torch.cat([preprocess_mask(mask_, res[0], res[1], device) for mask_ in binary_matrixes]) # torch.Size([10, 1, 64, 64])
    layouts_s = torch.bmm(layouts.view(layouts.size(0),-1,1), layouts.view(layouts.size(0),1,-1)).sum(0).unsqueeze(0).repeat(bsz,1,1) # torch.Size([1, 4096, 4096])
    reg_image_sizes = 1 - sizereg_image*layouts_s.sum(-1, keepdim=True)/(res[0]*res[1]) # torch.Size([1, 4096, 1])
    sreg_image_maps = layouts_s # torch.Size([1, 4096, 4096])
    
    layout_background = torch.cat([preprocess_mask(binary_matrixes[0], res[0], res[1], device)]) # torch.Size([1, 1, 64, 64])
    layouts_s_bakcground = torch.bmm(layout_background.view(layout_background.size(0),-1,1), layout_background.view(layout_background.size(0),1,-1)).sum(0).unsqueeze(0).repeat(bsz,1,1) # torch.Size([1, 4096, 4096])
    reg_image_sizes_bakcground = 1 - sizereg_image*layouts_s_bakcground.sum(-1, keepdim=True)/(res[0]*res[1]) # torch.Size([1, 4096, 1])
    
    layout_instance = torch.cat([preprocess_mask(mask_, res[0], res[1], device) for mask_ in binary_matrixes[1:]]) # torch.Size([8, 1, 64, 64])
    layouts_s_instance = torch.bmm(layout_instance.view(layout_instance.size(0),-1,1), layout_instance.view(layout_instance.size(0),1,-1)).sum(0).unsqueeze(0).repeat(bsz,1,1) # torch.Size([1, 4096, 4096])
    reg_image_sizes_instance = 1 - sizereg_image*layouts_s_instance.sum(-1, keepdim=True)/(res[0]*res[1])# torch.Size([1, 4096, 1])
    
    
    ###########################
    ###### prep for image to text creg ######
    ###########################
    pww_maps_background = torch.zeros(1, 512, res[0], res[1]).to(device) # torch.Size([1, 512, 64, 64])
    pww_maps_attribute = torch.zeros(1, 512, res[0], res[1]).to(device) # torch.Size([1, 512, 64, 64])
    pww_maps_instance = torch.zeros(1, 512, res[0], res[1]).to(device) # torch.Size([1, 512, 64, 64])
    ptt_maps = torch.eye(512, 512).unsqueeze(0).to(device) # torch.Size([1, 512, 512])
    for i in range(1,len(prompts)):
        wlen = text_input["attention_mask"][i].sum(dim=-1).item() - 1
        widx = text_input['input_ids'][i][:wlen]
        for j in range(512):
            try:
                if (text_input['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
                    ptt_maps[:, j:j+wlen, j:j+wlen] = 1
                    prompt_embeds[0][j:j+wlen] = prompt_embeds[i][:wlen]

                    if i == 1:
                        pww_maps_background[:,j:j+wlen,:,:] = layouts[0:1]
                    else:
                        attributet, instancet = attribute_instance(prompts[i])
                        text_input_t = pipe.tokenizer_2([attributet, instancet], padding="max_length", max_length= 512,truncation=True, return_length=False, return_overflowing_tokens=False, return_tensors="pt")
                        wlen_attribute =  text_input_t["attention_mask"][0].sum(dim=-1).item() - 1
                        index = -1
                        for k in range(512):
                            if (text_input['input_ids'][0][k:k+wlen_attribute] == text_input_t['input_ids'][0][:wlen_attribute]).sum() == wlen_attribute:
                                pww_maps_attribute[:,k:k+wlen_attribute,:,:] = layouts[i-1:i]
                                index = k
                                break
                        if index > 0:
                            pww_maps_instance[:, j:index,:,:] = layouts[i-1:i]
                            pww_maps_instance[:, index+wlen_attribute :j+wlen,:,:] = layouts[i-1:i]
                        else:
                            pww_maps_instance[:,j:j+wlen,:,:] = layouts[i-1:i]
                    break
            except:
                raise gr.Error(f"Please check -{prompts[i]}- is included in the full text !")
                return

    pww_maps_background = pww_maps_background.view(1,512, -1).permute(0,2,1).repeat(bsz,1,1) # torch.Size([1, 4096, 512])
    pww_maps_attribute = pww_maps_attribute.view(1,512, -1).permute(0,2,1).repeat(bsz,1,1) # torch.Size([1, 4096, 512])
    pww_maps_instance = pww_maps_instance.view(1,512, -1).permute(0,2,1).repeat(bsz,1,1) # torch.Size([1, 4096, 512])
    
    ###########################
    ###### prep for text to text sreg ######
    ###########################    
    reg_text_sizes = 1 - sizereg_text*ptt_maps.sum(-1, keepdim=True)/512 # torch.Size([1, 512, 1])
    sreg_text_maps = ptt_maps # torch.Size([1, 512, 512])
  
    
    if seed == -1:
        generator_s = torch.Generator("cpu")
    else:
        generator_s = torch.Generator("cpu").manual_seed(seed)
    
    images = pipe(num_images_per_prompt = bsz, output_type="pil", 
                    pooled_prompt_embeds = pooled_prompt_embedding,
                    prompt_embeds = prompt_embeds[0: 1], 
                    num_inference_steps=num_inference_steps, 
                    height = height, width = width, generator=generator_s).images[0]
    
    current_time = datetime.now().strftime('%d-%H-%M')
    save_dir = f'./result/{current_time}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"{master_prompt[:min(len(prompts[0])-1, 50)]} - {seed} - {t2treg} - {i2treg} - {num_mod_steps} - {num_inference_steps}.png")
    images.save(save_path)
    
    return([images])


#################################################
#################################################
### define the interface
with gr.Blocks(css=css) as demo:
    binary_matrixes = gr.State([])
    color_layout = gr.State([])
    gprompt = gr.State()
    gr.Markdown('''##Hierarchical and Step-Layer-Wise Tuning of Attention Specialty for Multi-Instance Synthesis in Diffusion Transformers''')
    gr.Markdown('''
    #### 😺 Instruction to generate images 😺 <br>
    (1) Create the image layout. <br>
    (2) Enter text prompt and label each segment. <br>
    (3) Check the generated images, and tune the hyperparameters if needed. <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - w<sup>c</sup> : The degree of attention modulation at cross-attention layers. <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - w<sup>s</sup> : The degree of attention modulation at self-attention layers. <br>
    ''')
    
    with gr.Row():
        with gr.Group(elem_id="main-image"):
            canvas_data = gr.JSON(value={}, visible=False)
            canvas = gr.HTML(canvas_html)
            button_run = gr.Button("(1) I've finished my sketch ! 😺", elem_id="main_button", interactive=True)
      
            prompts = []
            colors = []
            color_row = [None] * MAX_COLORS
            with gr.Column(visible=False) as post_sketch:
                general_prompt = gr.Textbox(label="Textual Description", value='')
                for n in range(MAX_COLORS):
                    if n == 0 :
                        with gr.Row(visible=False) as color_row[n]:
                            colors.append(gr.Image(label="background", type="pil", image_mode="RGB", width=100, height=100))
                            prompts.append(gr.Textbox(label="Prompt for the background (white region)", value=""))
                    else:
                        with gr.Row(visible=False) as color_row[n]:
                            colors.append(gr.Image(label="segment "+str(n), type="pil", image_mode="RGB", width=100, height=100))
                            prompts.append(gr.Textbox(label="Prompt for the segment "+str(n)))
                get_genprompt_run = gr.Button("(2) I've finished prompts and segment labeling ! 😺", elem_id="prompt_button", interactive=True)
            
            with gr.Column(visible=False) as tune_hyperparameter:
                with gr.Accordion("(3) Tune the hyperparameters", open=False):
                    num_inference_steps_ = gr.Slider(label="Number of inference steps", minimum=1, maximum=50, value=32, step=1)
                    num_mod_steps_ = gr.Slider(label="Number of tuning steps", minimum=0, maximum=50, value=16, step=1)
                    height_ = gr.Slider(label="Height", minimum=128, maximum=8192, value=1024, step=16)
                    width_ = gr.Slider(label="Width", minimum=128, maximum=8192, value=1024, step=16)
                    t2treg_ = gr.Slider(label="w\u1D9C--Degree of T2T attention modulation module", minimum=0, maximum=10., value=3.5, step=0.1)
                    i2treg_ = gr.Slider(label="w\u1D48--Degree of I2T attention modulation module", minimum=0, maximum=10., value=5.0, step=0.1)
                    i2ireg_ = gr.Slider(label="w\u1DA0--Degree of I2I attention modulation module", minimum=0, maximum=10., value=3.5, step=0.1)
                    decay_size_ = gr.Slider(label="Decay rate of attention modulation", minimum=0, maximum=10., value=3., step=0.1)
                    sizereg_image_ = gr.Slider(label="Degree of image-to-query area modulation", minimum=0, maximum=30., value=1.0, step=0.1)
                    sizereg_text_ = gr.Slider(label="Degree of text-to-query area modulation", minimum=0, maximum=50., value=4., step=0.1)
                    seed_ = gr.Slider(label="Seed", minimum=-1, maximum=999999999, value=-1, step=1)
                    
                final_run_btn = gr.Button("Finally ! Generate ! 😺")
                layout_path = gr.Textbox(label="layout_path", visible=False)
                all_prompts = gr.Textbox(label="all_prompts", visible=False)

                
        with gr.Column():
            out_image = gr.Gallery(label="Result", columns=2, height='auto')
            
    button_run.click(process_sketch, inputs=[canvas_data], outputs=[post_sketch, binary_matrixes, *color_row, *colors], js=get_js_colors, queue=False)
    
    get_genprompt_run.click(process_prompts, inputs=[general_prompt, *prompts], outputs=[tune_hyperparameter, gprompt], queue=False)
    
    final_run_btn.click(process_generation, inputs=[binary_matrixes, seed_, num_inference_steps_, num_mod_steps_, height_, width_, t2treg_, i2treg_,i2ireg_, sizereg_image_, sizereg_text_, decay_size_, general_prompt, *prompts], outputs=out_image)
 
    gr.Examples(
        examples=[[val_layout + '0.png',
                   '***'.join([val_prompt[0]['textual_condition']] + val_prompt[0]['segment_descriptions']), 945554353],
                  [val_layout + '1.png',
                   '***'.join([val_prompt[1]['textual_condition']] + val_prompt[1]['segment_descriptions']), 315680852],
                  [val_layout + '2.png',
                   '***'.join([val_prompt[2]['textual_condition']] + val_prompt[2]['segment_descriptions']), 128785431]],
        inputs=[layout_path, all_prompts, seed_],
        outputs=[post_sketch, binary_matrixes, *color_row, *colors, *prompts, tune_hyperparameter, general_prompt, seed_],
        fn=process_example,
        run_on_click=True,
        label='😺 Examples 😺',
    ) 
   
    
    demo.load(None, None, None, js=load_js)
    
demo.launch(server_name="0.0.0.0")