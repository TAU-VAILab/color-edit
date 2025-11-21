import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import abc
from typing import Optional, Union, Tuple, List, Callable, Dict

from torchvision.utils import save_image
from einops import rearrange, repeat

class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

class AttentionStore(AttentionBase):
    
        
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 64 ** 2:
            if is_cross:
                self.cross_attns_step.append(attn)
            else:
                self.self_attns_step.append(attn)
            if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
                self.step_store[key].append(attn)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
    
    
    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1                            
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]

        self.self_attns_step.clear()
        self.cross_attns_step.clear()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention
    
    def __init__(self, res=[32], min_step=0, max_step=1000,low_resource=True):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.valid_steps = 0

        self.self_attns = []  # store the all attns
        self.cross_attns = []

        self.self_attns_step = []  # store the attns in each step
        self.cross_attns_step = []
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.low_resource = low_resource


class SelfAttentionControlEditCic(AttentionStore, abc.ABC):
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.low_resource else 0

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:# or is_cross:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.after_step()
            # self.between_steps()
        return attn
    
    def step_callback(self, x_t):
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet,is_cross=False):
        if att_replace.shape[2] <= 32 ** 2:# or is_cross:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
                      
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                pass
            # if is_cross:
            #     # replace cross attention map of target prompts with source prompt(last one on batch)
            #     if attn_repalce.shape[0]>=2:# and self.cur_step<25:
            #         attn[1:-1] = self.replace_self_attention(attn_repalce[-1], attn_repalce[0:-1], place_in_unet, is_cross)
            #     else:
            #         pass
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 64 ** 2:
            # if is_cross:
            #     self.cross_attns_step.append(attn)
            # else:
            #     self.self_attns_step.append(attn)
            if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
                self.step_store[key].append(attn)
        
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]], low_resource: bool):
        super(SelfAttentionControlEditCic, self).__init__(low_resource=low_resource)
        self.batch_size = len(prompts)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_steps = num_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        # self.final_step_store = self.get_empty_store()
    
    # def reset(self):
    #     super(SelfAttentionControlEditCic, self).reset()
    #     self.final_step_store = self.get_empty_store()
    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            # if len(self.self_attns) == 0:
            #     self.self_attns = self.self_attns_step
            #     self.cross_attns = self.cross_attns_step
            # else:
            #     for i in range(len(self.self_attns)):
            #         self.self_attns[i] += self.self_attns_step[i]
            #         self.cross_attns[i] += self.cross_attns_step[i]
                            
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
        
        self.self_attns_step.clear()
        self.cross_attns_step.clear()
        # maintain last setp step_store attentions - only for low_resource=False
        if self.cur_step < self.num_steps: 
            self.step_store = self.get_empty_store()


class SelfCrossAttentionControlColorEdit(AttentionStore, abc.ABC):
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.low_resource else 0

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:# or is_cross:
            # if self.low_resource:
            #     attn = self.forward(attn, is_cross, place_in_unet)
            # else:
            h = attn.shape[0]
            if h==8: # 1 batch
                key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"           
                if attn.shape[1] <= 64 ** 2:
                    if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
                        self.attention_store_step[key].append(attn)
            else:
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            if h!=8:
                self.cur_step += 1
                self.after_step()
            # self.between_steps()
        return attn
    
    def get_average_attention_step(self):
        average_attention = self.attention_store_step
        return average_attention
    
    def step_callback(self, x_t):
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2 :#and not (self.cur_att_layer//2 <= 9  and self.cur_att_layer//2 >= 4):#or not self.low_resource:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    def replace_cross_attention(self, attn_base, att_replace, place_in_unet, is_cross=False):
        attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        return att_replace

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                # alpha_words = self.cross_replace_alpha[self.cur_step]
                # attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                # replace
                # attn_refer = self.reference_cross_attn['cross'][self.cur_att_layer//2]
                # h = attn_refer.shape[0] // (self.batch_size)
                # attn_refer = attn_refer.reshape(self.batch_size, h, *attn_refer.shape[1:])
                # attn[1:] = self.replace_cross_attention(attn_refer[1], attn_repalce, place_in_unet) #attn_repalce_new
                pass
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"           
        if attn.shape[1] <= 64 ** 2:
            # if is_cross :
            #     self.cross_attns_step.append(attn.to('cpu'))
            # else:
            #     self.self_attns_step.append(attn.to('cpu'))
            if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
                self.step_store[key].append(attn.to('cpu'))
                # self.attention_store_step[key].append(attn)
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 self_replace_steps: Union[float, Tuple[float, float]], low_resource: bool,
                 reference_attention_store: dict):
        super(SelfCrossAttentionControlColorEdit, self).__init__(low_resource=low_resource)
        self.batch_size = len(prompts)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_steps = num_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.reference_attention_store = reference_attention_store 
        self.attention_store_step = self.get_empty_store()


def regiter_attention_editor_diffusers(model, editor: AttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count


def register_attention_control_FPE(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            x = hidden_states
            context = encoder_hidden_states
            mask = attention_mask
            batch_size, sequence_length, dim = x.shape
            # if sequence_length > 16**2 :
            #     if context is not None:
            #         context[4] = context[3]
            #     else:
            #         hidden_states[4] = hidden_states[3]
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
            return to_out(out)

        return forward

    def register_recr(net_, count, place_in_unet):
        # print(net_.__class__.__name__)
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    # print(sub_nets)
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
    

def register_attention_control_ColorEdit(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            x = hidden_states
            context = encoder_hidden_states
            mask = attention_mask
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.head_to_batch_dim(q)
            k = self.head_to_batch_dim(k)
            v = self.head_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.batch_to_head_dim(out)
            return to_out(out)

        return forward

    def register_recr(net_, count, place_in_unet):
        # print(net_.__class__.__name__)
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    # print(sub_nets)
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count