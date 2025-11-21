import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nltk
from typing import Tuple, List
from PIL import Image
from sklearn.cluster import KMeans
from coloredit_code.controllers import AttentionStore, SelfCrossAttentionControlColorEdit

sys.path.append('enviroment/src/local_prompt_mixing/')
from src.attention_utils import aggregate_attention, text_under_image

def aggregate_attention_syngen(controller: SelfCrossAttentionControlColorEdit, res: int, from_where: List[str], is_cross: bool, select: int, prompts):
    out = []
    attention_maps = controller.get_average_attention_step()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out

def aggregate_attention_refer(controller: SelfCrossAttentionControlColorEdit, res: int, from_where: List[str], is_cross: bool, select: int, prompts):
    out = []
    attention_maps = {key: [item / controller.num_steps for item in controller.reference_attention_store[key]] for key in
                             controller.reference_attention_store}
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out

def mask_unique(mask1,mask2):
    overlapping_mask = torch.logical_and(mask1,mask2) # overlapping mask
    new_mask1 = mask1*torch.logical_xor(mask1, overlapping_mask) # remove overlapping mask
    new_mask2 = mask2*torch.logical_xor(mask2, overlapping_mask) # remove overlapping mask
    if torch.sum(mask1) != 0 and torch.sum(mask2) != 0 :
        if torch.sum(overlapping_mask)/torch.sum(mask1) > torch.sum(overlapping_mask)/torch.sum(mask2):
            new_mask1 = mask1 # assign overlapping mask to the object which cotains the most of it
        else:
            new_mask2 = mask2
    return new_mask1, new_mask2

class SegmentorCic:

    def __init__(self, controller, prompts, num_segments, background_segment_threshold, res=32, background_nouns=[]):
        self.controller = controller
        self.prompts = prompts
        self.num_segments = num_segments
        self.background_segment_threshold = background_segment_threshold
        self.resolution = res
        self.background_nouns = background_nouns

        self.self_attention = aggregate_attention_refer(controller, res=32, from_where=("up", "down"), prompts=prompts,
                                             is_cross=False, select=len(prompts) - 1)
        self.cross_attention = aggregate_attention_refer(controller, res=16, from_where=("up", "down"), prompts=prompts,
                                              is_cross=True, select=len(prompts) - 1)
        tokenized_prompt = nltk.word_tokenize(prompts[-1])
        self.nouns = [(i, word) for (i, (word, pos)) in enumerate(nltk.pos_tag(tokenized_prompt)) if pos[:2] == 'NN']

    def __call__(self, *args, **kwargs):
        clusters = self.cluster()
        cluster2noun = self.cluster2noun(clusters)
        return cluster2noun

    def cluster(self):
        np.random.seed(1)
        resolution = self.self_attention.shape[0]
        attn = self.self_attention.cpu().numpy().reshape(resolution ** 2, resolution ** 2)
        kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(attn)
        clusters = kmeans.labels_
        clusters = clusters.reshape(resolution, resolution)
        return clusters

    def cluster2noun(self, clusters):
        result = {}
        nouns_indices = [index for (index, word) in self.nouns]
        nouns_maps = self.cross_attention.cpu().numpy()[:, :, [i + 1 for i in nouns_indices]]
        normalized_nouns_maps = np.zeros_like(nouns_maps).repeat(2, axis=0).repeat(2, axis=1)
        for i in range(nouns_maps.shape[-1]):
            curr_noun_map = nouns_maps[:, :, i].repeat(2, axis=0).repeat(2, axis=1)
            normalized_nouns_maps[:, :, i] = (curr_noun_map - np.abs(curr_noun_map.min())) / curr_noun_map.max()
        for c in range(self.num_segments):
            cluster_mask = np.zeros_like(clusters)
            cluster_mask[clusters == c] = 1
            score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))]
            scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
            result[c] = self.nouns[np.argmax(np.array(scores))] if max(scores) > self.background_segment_threshold else "BG"
        return result

    def get_background_mask(self, obj_token_index):
        clusters = self.cluster()
        colors = np.array(
                        [(228, 26, 28),   # Red 0
                        (18, 82, 131),   # Blue 1  
                        (50, 186, 88),   # Green 2 
                        (127, 30, 135),  # Purple 3 
                        (222, 124, 9),   # Orange 4 
                        (244, 229, 43),  # Yellow 5 
                        (65, 133, 189),  # Cyan 6 
                        (158, 73, 37)    # Brown 7
                        ])
        cluster2noun = self.cluster2noun(clusters)
        mask = clusters.copy()
        obj_segments = [c for c in cluster2noun if cluster2noun[c][0] == obj_token_index - 1]
        background_segments = [c for c in cluster2noun if cluster2noun[c] == "BG" or cluster2noun[c][1] in self.background_nouns]
        for c in range(self.num_segments):
            if c in background_segments and c not in obj_segments:
                mask[clusters == c] = 0
            else:
                mask[clusters == c] = 1
        return mask, colors[clusters]/255




def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], prompts, tokenizer, select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, prompts)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    return create_images(np.stack(images, axis=0))


def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str], prompts,
                             max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select, prompts).numpy().reshape(
        (res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    return create_images(np.concatenate(images, axis=1))


def create_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img

def write_maps(image):
    image.save('./tmp/attn.jpg')

def show_cross_self_attention_maps(controller,res={'cross':16,'self':32},from_where=("up", "down"),prompts=[''], tokenizer=None ,select=0, out_dir='tmp',diff_step=50,prefix=''):
    if diff_step in [25,40,50]: # to save time as self attn map require SVD decomp.
        print("working on cross attention")
        if prefix != '':
            prefix = prefix + '_'
        cross_attn = show_cross_attention(controller, res['cross'],from_where=list(from_where), prompts=prompts, tokenizer=tokenizer, select=select)
        cross_attn.save(os.path.join(out_dir, f'{prefix}cross_attn_{diff_step}_{prompts[select]}.jpg'))
    
        print("working on self attention")
        self_attn = show_self_attention_comp(controller, res['self'],from_where=list(from_where), prompts=prompts, max_com=10, select=select)
        self_attn.save(os.path.join(out_dir, f'{prefix}self_attn_{diff_step}_{prompts[select]}.jpg'))
