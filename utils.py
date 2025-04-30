import torch
import base64
import gradio as gr
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import io
import random
import os
import spacy
from datetime import datetime


def attribute_instance(sentence):
    if sentence == "a white goat":
        return "white", "goat"
    nlp = spacy.load('en_core_web_lg')
    doc = nlp(sentence)
    
    noun_chunks = list(doc.noun_chunks)
    if not noun_chunks:
        return "", sentence
    
    target_chunk = noun_chunks[0]
    root = target_chunk.root
    
    adjs = [
        token for token in target_chunk 
        if token.dep_ in ('amod', 'conj', 'compound')  
        and token.head == root 
        and token.pos_ in ('ADJ', 'NOUN')
    ]
    

    if not adjs:
        return "", target_chunk.text.strip()


    adj_bounds = [(t.idx, t.idx + len(t.text)) for t in adjs]
    attr_start = min(s for s, _ in adj_bounds)
    attr_end = max(e for _, e in adj_bounds)
    

    noun_start = root.idx
    noun_end = noun_start + len(root.text)
    

    instance_start = noun_start
    instance_end = noun_end
    
    while instance_start > target_chunk.start_char:
        prev_char_pos = instance_start - 1
        if prev_char_pos < attr_end and not sentence[prev_char_pos].isspace():
            instance_start -= 1
        else:
            break

    while instance_end < target_chunk.end_char:
        next_char_pos = instance_end
        if next_char_pos > attr_start and not sentence[next_char_pos].isspace():
            instance_end += 1
        else:
            break
    
    if instance_start < attr_end:
        instance_start = noun_start
    

    attribute_text = sentence[attr_start:attr_end].strip()
    instance_text = sentence[instance_start:instance_end].strip()
    
    return attribute_text, instance_text

MAX_COLORS = 12


def create_binary_matrix(img_arr, target_color):
    mask = np.all(img_arr == target_color, axis=-1)
    binary_matrix = mask.astype(int)
    return binary_matrix

def preprocess_mask(mask_, h, w, device):
    mask = np.array(mask_)
    mask = mask.astype(np.float32)
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(device)
    mask = torch.nn.functional.interpolate(mask, size=(h, w), mode='nearest')
    return mask

def process_sketch(canvas_data):
    binary_matrixes = []
    base64_img = canvas_data['image']
    image_data = base64.b64decode(base64_img.split(',')[1])
    image = Image.open(BytesIO(image_data)).convert("RGB")
    current_time = datetime.now().strftime('%d-%H-%M')
    image.save(f"./dataset/{current_time}.png", "PNG") # save the segment picture in png format.
    im2arr = np.array(image)
    colors = [tuple(map(int, rgb[4:-1].split(','))) for rgb in canvas_data['colors']]
    colors_fixed = []
    
    r, g, b = 255, 255, 255
    binary_matrix = create_binary_matrix(im2arr, (r,g,b))
    binary_matrixes.append(binary_matrix)
    binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
    colored_map = binary_matrix_*(r,g,b) + (1-binary_matrix_)*(50,50,50)
    colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))
    for color in colors:
        r, g, b = color
        if any(c != 255 for c in (r, g, b)):
            binary_matrix = create_binary_matrix(im2arr, (r,g,b))
            binary_matrixes.append(binary_matrix)
            binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
            colored_map = binary_matrix_*(r,g,b) + (1-binary_matrix_)*(50,50,50)
            colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))
            
    visibilities = []
    colors = []
    for n in range(MAX_COLORS):
        visibilities.append(gr.update(visible=False))
        colors.append(gr.update())
    for n in range(len(colors_fixed)):
        visibilities[n] = gr.update(visible=True)
        colors[n] = colors_fixed[n]
    
    return [gr.update(visible=True), binary_matrixes, *visibilities, *colors]


def process_prompts(textual_prompt, *seg_prompts):
    for seg in seg_prompts:
        if seg not in textual_prompt:
            #raise gr.Error(f"Please check whether -{seg}- prompt is included in the full text!")
            return
            
    return [gr.update(visible=True), gr.update(value = textual_prompt)]


def process_example(layout_path, all_prompts, seed_):
    
    all_prompts = all_prompts.split('***')
    
    binary_matrixes = []
    colors_fixed = []
    
    im2arr = np.array(Image.open(layout_path))[:,:,:3]
    unique, counts = np.unique(np.reshape(im2arr,(-1,3)), axis=0, return_counts=True)
    sorted_idx = np.argsort(-counts)
    
    binary_matrix = create_binary_matrix(im2arr, (255, 255, 255))
    binary_matrixes.append(binary_matrix)
    binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
    colored_map = binary_matrix_*(255,255,255) + (1-binary_matrix_)*(50,50,50)
    colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))
    for i in range(len(all_prompts)-1):
        r, g, b = unique[sorted_idx[i]]
        if any(c != 255 for c in (r, g, b)) and any(c != 0 for c in (r, g, b)):
            binary_matrix = create_binary_matrix(im2arr, (r,g,b))
            binary_matrixes.append(binary_matrix)
            binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
            colored_map = binary_matrix_*(r,g,b) + (1-binary_matrix_)*(50,50,50)
            colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))
    visibilities = []
    colors = []
    prompts = []
    for n in range(MAX_COLORS):
        visibilities.append(gr.update(visible=False))
        colors.append(gr.update())
        prompts.append(gr.update())
        
    for n in range(len(colors_fixed)):
        visibilities[n] = gr.update(visible=True)
        colors[n] = colors_fixed[n]
        prompts[n] = all_prompts[n+1]
    
    return [gr.update(visible=True), binary_matrixes, *visibilities, *colors, *prompts,
            gr.update(visible=True), gr.update(value=all_prompts[0]), int(seed_)]

def set_seed_default(seed_int = 4562489):
    np.random.seed(seed_int)
    random.seed(seed_int)
    torch.manual_seed(seed_int)
    torch.cuda.manual_seed(seed_int)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
