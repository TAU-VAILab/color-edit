import os
import torch
import cv2
from cv2 import dilate
import numpy as np
import sys

import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything
from diffusers import StableDiffusionPipeline

from coloredit_code.utils import aggregate_attention_syngen, aggregate_attention_refer, show_cross_self_attention_maps, mask_unique, SegmentorCic
from coloredit_code.losses import symmetric_kl

sys.path.append('enviroment/src/')
sys.path.append('enviroment/src/MasaCtrl/')
sys.path.append('enviroment/src/local_prompt_mixing/')
from masactrl.diffuser_utils import MasaCtrlPipeline
from local_prompt_mixing.src.attention_based_segmentation import Segmentor



class ColorEditDiffsuionPipeline(MasaCtrlPipeline):

    # exctract the cross attention maps of the color and manipulate them using the masks to make sure they don't overlapp
    def get_segmentation_mask(self, controller,prompts,
                              num_segments,background_segment_threshold, height, width,
                              relative_dict,relative_dict_mask,target_prompt_token_indicies):
        relative_dict_mask_seperated = {}
        total_mask = torch.zeros((1,1,height//8, width//8)).cuda()
        for idx, (object_name,background_nouns) in enumerate(relative_dict.items()):
            target_object_index_in_target_prompt = prompts[-1].split(' ').index(object_name) + 1
            if len(prompts)==2:
                object_mask = relative_dict_mask[object_name]
                seg_cross = Segmentor(controller,
                                    prompts,
                                    num_segments,
                                    background_segment_threshold,
                                    background_nouns=background_nouns)
                object_cross_attn_map = seg_cross.cross_attention[:,:,target_prompt_token_indicies[idx][-1] -1] 
            else:
                seg = Segmentor(controller,
                                prompts,
                                num_segments,
                                background_segment_threshold,
                                background_nouns=background_nouns)
                object_mask = seg.get_background_mask(target_object_index_in_target_prompt)
                object_cross_attn_map = seg.cross_attention[:,:,target_object_index_in_target_prompt-1] # color cross attn map

            tmp_map = (object_cross_attn_map - object_cross_attn_map.min())
            object_cross_attn_map =  tmp_map / tmp_map.max()      
            mask = object_mask.astype(np.bool8)
            mask = torch.from_numpy(mask).float().cuda()
            shape = (1, 1, mask.shape[0], mask.shape[1])
            if not (height//8 == mask.shape[0] and width//8 == mask.shape[1]):  
                mask = torch.nn.Upsample(size=(height//8, width//8), mode='nearest')(mask.view(shape))
                mask_eroded = dilate(mask.cpu().numpy()[0, 0], np.ones((3, 3), np.uint8), iterations=1)
            else:
                mask_eroded = mask.cpu().numpy()
            mask = torch.from_numpy(mask_eroded).float().cuda().view(1, 1, height//8, width//8)
            shape = (1, 1, object_cross_attn_map.shape[0], object_cross_attn_map.shape[1])
            
            object_cross_attn_map = transforms.functional.resize(object_cross_attn_map.view(shape), (height//8, width//8),
                            interpolation=transforms.InterpolationMode.BICUBIC,
                            antialias=True).clamp(0,1)
            object_cross_attn_map = object_cross_attn_map.float().cuda().view(1, 1, height//8, width//8)
            object_cross_attn_map_masked = object_cross_attn_map
            relative_dict_mask_seperated[object_name] = [mask, object_cross_attn_map_masked]

            total_mask += mask
        
        
        if len(relative_dict_mask_seperated.keys())>=2:
            three_objects_prompt = False
            if len(relative_dict_mask_seperated.keys())==3:
                first_obj, second_obj, third_obj = list(relative_dict_mask_seperated.keys())
                three_objects_prompt = True
            else:
                first_obj, second_obj = list(relative_dict_mask_seperated.keys())
            
            # find unique mask in 2 object prompts
            mask1, attn_map1 = relative_dict_mask_seperated[first_obj]
            mask2, attn_map2 = relative_dict_mask_seperated[second_obj]
            mask1, mask2 = mask_unique(mask1,mask2)
            
            # find unique mask in 3 object prompts
            if three_objects_prompt:
                mask3, attn_map3 = relative_dict_mask_seperated[third_obj]
                mask1, mask3 = mask_unique(mask1,mask3)
                mask2, mask3 = mask_unique(mask2,mask3)

            # unique cross attention map
            new_attn_map1 = mask1*attn_map1
            new_attn_map2 = mask2*attn_map2
            
            #normalize
            tmp_map = (new_attn_map1 - new_attn_map1.min())
            new_attn_map1 =  tmp_map / tmp_map.max()     
            
            #normalize
            tmp_map = (new_attn_map2 - new_attn_map2.min())
            new_attn_map2 =  tmp_map / tmp_map.max()

            relative_dict_mask_seperated[first_obj] = [mask1, new_attn_map1]
            relative_dict_mask_seperated[second_obj] = [mask2, new_attn_map2] 
            
            if three_objects_prompt:
                # unique cross attention map
                new_attn_map3 = mask3*attn_map3
            
                #normalize
                tmp_map = (new_attn_map3 - new_attn_map3.min())
                new_attn_map3 =  tmp_map / tmp_map.max()
                
                relative_dict_mask_seperated[third_obj] = [mask3, new_attn_map3] 
            
        return relative_dict_mask_seperated, total_mask.clamp(0,1)
            
    # attention loss function      
    def kl_cross_attn_update(self, latents, diff_step,t, text_embeddings, controller, prompts, simplified_prompt,
                             simplified_prompt_token_indicies,target_prompt_token_indicies,attention_loss_iters=10,
                             attention_loss_weight=20,attnetion_loss_stoping_step=35,attnetion_symmetric_kl_bottom_limit=0.1):
        
        with torch.enable_grad():
            loss = 0.0
            loss_value = 0.0
            prev_not_run=False
            attention_maps_refer = aggregate_attention_refer(controller, res=16, from_where=("up", "down", "mid"), prompts=['',simplified_prompt],
                                is_cross=True, select=1)
            attention_maps_refer = attention_maps_refer.reshape(-1,attention_maps_refer.shape[-1])*100
            attention_maps_refer = attention_maps_refer      #.to('cuda')
            for i in range(attention_loss_iters):
                if prev_not_run:
                    continue
                latents = latents.clone().detach().requires_grad_(True)
                controller.attention_store_step =  {"down_cross": [], "mid_cross": [], "up_cross": [],
                    "down_self": [], "mid_self": [], "up_self": []}

                self.unet(latents, t, encoder_hidden_states=text_embeddings, return_dict=False)[0]
                self.unet.zero_grad()
                
                attention_maps = aggregate_attention_syngen(controller, res=16, from_where=("up", "down", "mid"), prompts=prompts[-1:],
                                                            is_cross=True, select=0)
                del controller.attention_store_step
                torch.cuda.empty_cache()

                attention_maps = attention_maps.reshape(-1,attention_maps.shape[-1])*100
                
                simplified_first_object_idx = simplified_prompt_token_indicies[0]
                target_first_object_idx = target_prompt_token_indicies[0]
                               
                # for token_idx in target_first_idx:
                first_color_skl = symmetric_kl(attention_maps[:,target_first_object_idx[0]],attention_maps_refer[:,simplified_first_object_idx[0]])
                if first_color_skl<attnetion_symmetric_kl_bottom_limit:
                    first_color_skl=0
                first_object_skl = symmetric_kl(attention_maps[:,target_first_object_idx[1]],attention_maps_refer[:,simplified_first_object_idx[0]])
                if first_object_skl<attnetion_symmetric_kl_bottom_limit:
                    first_object_skl=0

                second_color_skl=0
                second_object_skl=0
                third_color_skl=0
                third_object_skl=0
                if len(simplified_prompt_token_indicies)>=2:
                    # second object
                    simplified_second_object_idx = simplified_prompt_token_indicies[1]
                    target_second_object_idx = target_prompt_token_indicies[1]
                    second_color_skl = symmetric_kl(attention_maps[:,target_second_object_idx[0]],attention_maps_refer[:,simplified_second_object_idx[0]])
                    if second_color_skl<attnetion_symmetric_kl_bottom_limit:
                        second_color_skl=0
                    second_object_skl = symmetric_kl(attention_maps[:,target_second_object_idx[1]],attention_maps_refer[:,simplified_second_object_idx[0]])
                    if second_object_skl<attnetion_symmetric_kl_bottom_limit:
                        second_object_skl=0
                        
                    # third object if exists
                    if len(simplified_prompt_token_indicies)==3:
                        simplified_third_object_idx = simplified_prompt_token_indicies[2]
                        target_third_object_idx = target_prompt_token_indicies[2]
                        third_color_skl = symmetric_kl(attention_maps[:,target_third_object_idx[0]],attention_maps_refer[:,simplified_third_object_idx[0]])
                        if third_color_skl<attnetion_symmetric_kl_bottom_limit:
                            third_color_skl=0
                        third_object_skl = symmetric_kl(attention_maps[:,target_third_object_idx[1]],attention_maps_refer[:,simplified_third_object_idx[0]])
                        if third_object_skl<attnetion_symmetric_kl_bottom_limit:
                            third_object_skl=0                        

                loss = first_color_skl + first_object_skl + second_color_skl + second_object_skl + third_color_skl + third_object_skl
                
                # Perform gradient update
                if loss<=0.0:
                    prev_not_run=True
                    
                if diff_step < attnetion_loss_stoping_step and loss>0.0:
                    if loss != 0:
                        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
                        latents = latents - attention_loss_weight * grad_cond
                        del grad_cond
                
                del attention_maps
                torch.cuda.empty_cache()
        
        print(f"diff_step {diff_step} | Loss: {loss:0.4f}, first_color_skl: {first_color_skl}, first_object_skl: {first_object_skl},\
              second_color_skl: {second_color_skl}, second_object_skl: {second_object_skl}, third_color_skl: {third_color_skl}, third_object_skl: {third_object_skl}")
        
        del loss_value
        del attention_maps_refer
        torch.cuda.empty_cache()
        controller.attention_store_step =  {"down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self": [], "mid_self": [], "up_self": []}
        return latents

    # create object-mask dict
    def create_relative_dict(self,
                             color_object_dict,
                             controller,
                             simplified_prompt,
                             num_segments,
                             background_segment_threshold,
                             out_dir,
                             SAM_mask_dict,
                             height,
                             width,
                             debug=True):
        relative_dict = {}
        relative_dict_mask = {}
        for object_name in color_object_dict.keys():
            curr_objects = list(color_object_dict.keys())
            curr_objects.remove(object_name)
            relative_dict[object_name] = curr_objects
        # use segmentor
        if not SAM_mask_dict:
            for idx, (object_name,background_nouns) in enumerate(relative_dict.items()):
                print('using SegmentorCic')
                seg = SegmentorCic(controller,
                                    ['',simplified_prompt],
                                    num_segments,
                                    background_segment_threshold,
                                    background_nouns=background_nouns)
                target_object_index_in_simplified_prompt = simplified_prompt.split(' ').index(object_name) + 1
                object_mask, seg_mask = seg.get_background_mask(target_object_index_in_simplified_prompt)
                relative_dict_mask[object_name] = object_mask
                if debug:
                    save_image(torch.squeeze(torch.Tensor(object_mask)), os.path.join(out_dir,f'original_mask_{object_name}.png'))
            if debug:
                save_image(torch.Tensor(seg_mask).permute(2,0,1), os.path.join(out_dir, f"output_seg.png"))
        # use SAM
        else:
            for object_name, mask_path in SAM_mask_dict.items():        # Reads a file using pillow
                PIL_mask = Image.open(mask_path)
                object_mask = torch.Tensor(transforms.PILToTensor()(PIL_mask)/255).float()#dtype='int32')
                object_mask = torch.squeeze(F.interpolate(object_mask[:3].unsqueeze_(0), (512, 512)),dim=0)
                object_mask_downsmapled = torch.nn.functional.max_pool2d(object_mask, kernel_size=8, stride=8)[0,:,:]
                relative_dict_mask[object_name] = np.array(object_mask_downsmapled.numpy(),dtype='int32')
                if debug:
                    save_image(torch.squeeze(torch.Tensor(object_mask_downsmapled)), os.path.join(out_dir,f'original_mask_{object_name}.png'))
        return relative_dict, relative_dict_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        batch_size=1,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        latents=None,
        unconditioning=None,
        neg_prompt=None,
        ref_intermediate_latents=None,
        return_intermediates=False,
        simplified_prompt='',
        simplified_prompt_token_indicies=[],
        target_prompt_token_indicies=[],
        num_segments=5,
        background_segment_threshold=0.3,
        background_blend_timestep=-1,
        background_blend_timestep_start=51,
        controller=None,
        attention_loss_iters=10,
        attention_loss_weight=20,
        attnetion_loss_stoping_step=35,
        attnetion_symmetric_kl_bottom_limit=0.1,
        color_loss_weight=1.5,
        color_loss_starting_step=25,
        debug=False,
        **kwds):
        
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if isinstance(prompt, list):
            batch_size = len(prompt)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embeddings :", text_embeddings.shape)
        if kwds.get("dir"):
            dir = text_embeddings[-2] - text_embeddings[-1]
            u, s, v = torch.pca_lowrank(dir.transpose(-1, -2), q=1, center=True)
            text_embeddings[-1] = text_embeddings[-1] + kwds.get("dir") * v
            print(u.shape)
            print(v.shape)

        # define initial latents
        latents_shape = (batch_size, self.unet.in_channels, height//8, width//8)
        if latents is None:
            latents = torch.randn(latents_shape, device=DEVICE)
        else:
            assert latents.shape == latents_shape, f"The shape of input latent tensor {latents.shape} should equal to predefined one."

        # unconditional embedding for classifier free guidance
        if guidance_scale > 1.:
            max_length = text_input.input_ids.shape[-1]
            if neg_prompt:
                uc_text = neg_prompt
            else:
                uc_text = ""
            unconditional_input = self.tokenizer(
                [uc_text] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embeddings = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("latents shape: ", latents.shape)
        # iterative sampling
        self.scheduler.set_timesteps(num_inference_steps)
        if return_intermediates:
            latents_list = [latents]
            pred_x0_list = [latents]
        total_mask = torch.zeros((1,1,height//8, width//8)).to(DEVICE)
        self.diff_step = 0
        
        # controller != None only when running with target prompt - we have the attantion maps from the simplified prompt run
        # we create mask dict for each object using SAM or with segmentor
        if controller != None: 
            relative_dict, relative_dict_mask = self.create_relative_dict(kwds['color_object_dict'],
                                                                          controller,
                                                                          simplified_prompt,
                                                                          num_segments,
                                                                          background_segment_threshold,
                                                                          kwds["out_dir"], 
                                                                          kwds["SAM_mask_dict"],
                                                                          height=height,
                                                                          width=width,
                                                                          debug=True)
        # Diffusion loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampler")):
            if ref_intermediate_latents is not None:
                # note that the batch_size >= 2
                latents_ref = ref_intermediate_latents[-1 - i]
                _, latents_cur = latents.chunk(2)
                latents = torch.cat([latents_ref, latents_cur])
            
            # attention loss
            if ((controller != None) and self.diff_step< attnetion_loss_stoping_step):
                if kwds['use_attention_loss']:
                    latents_cur = self.kl_cross_attn_update(latents_cur,
                                                            self.diff_step,
                                                            t,
                                                            text_embeddings[-1:],
                                                            controller,
                                                            prompts=prompt, 
                                                            simplified_prompt=simplified_prompt,
                                                            simplified_prompt_token_indicies=simplified_prompt_token_indicies,
                                                            target_prompt_token_indicies=target_prompt_token_indicies,
                                                            attention_loss_iters=attention_loss_iters,
                                                            attention_loss_weight=attention_loss_weight,
                                                            attnetion_loss_stoping_step=attnetion_loss_stoping_step,
                                                            attnetion_symmetric_kl_bottom_limit=attnetion_symmetric_kl_bottom_limit)
                    latents = torch.cat([latents_ref, latents_cur])
                
            if guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents
            if unconditioning is not None and isinstance(unconditioning, list):
                _, text_embeddings = text_embeddings.chunk(2)
                text_embeddings = torch.cat([unconditioning[i].expand(*text_embeddings.shape), text_embeddings]) 
            
            # predict tghe noise
            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embeddings).sample
            if guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            
            # compute the previous noise sample x_t -> x_t-1 
            latents, pred_x0 = self.step(noise_pred, t, latents)
            
            # In the target prompt run we want to use the color cross attention maps for the colors loss thus we extract them here                         
            if ((self.diff_step>=color_loss_starting_step or 
                (self.diff_step>=background_blend_timestep_start and self.diff_step <= background_blend_timestep)) and (controller!=None)) :
                relative_dict_mask_seperated, total_mask = self.get_segmentation_mask(controller,
                                                                                      prompt,
                                                                                      num_segments,
                                                                                      background_segment_threshold,
                                                                                      height,
                                                                                      width,
                                                                                      relative_dict,
                                                                                      relative_dict_mask,
                                                                                      target_prompt_token_indicies=target_prompt_token_indicies)
                # save the masks and cross attention maps for debug
                if self.diff_step in [1,color_loss_starting_step,20,40,49]:
                    for object_name,masks in relative_dict_mask_seperated.items():
                        save_image(masks[0], os.path.join(kwds['out_dir'], f"mask_{self.diff_step}_{object_name}.png"))
                        save_image(masks[1], os.path.join(kwds['out_dir'], f"cross_attn_{self.diff_step}_{object_name}.png"))
                    save_image(torch.Tensor(total_mask), os.path.join(kwds['out_dir'], f"total_mask_{self.diff_step}.png"))
                
            # color loss
            if (self.diff_step>=color_loss_starting_step) and (controller!=None):
                if kwds['use_color_loss']:
                    print("running color loss")
                    if latents.shape[0]==3:
                        latents_recon, latents_cur, latents_ref = latents.chunk(3)
                    elif latents.shape[0]==2:
                        latents_recon, latents_cur = latents.chunk(2)
                    else:
                        latents_cur = latents
                    loss_total = 0.
                    with torch.enable_grad():
                        if not latents_cur.requires_grad:
                            latents_cur.requires_grad = True
                        imgs = self.latent2image_grad(latents_cur)
                        imgs = (imgs / 2 + 0.5).clamp(0, 1)
                        flag = False
                        for object_name, masks in relative_dict_mask_seperated.items():
                            if self.diff_step >=51:
                                mask_for_color = masks[0] 
                            else:
                                mask_for_color = masks[1]
                            rgb_val = kwds['color_object_dict'][object_name]
                            mask_img = transforms.functional.resize(mask_for_color, (512, 512),
                                interpolation=transforms.InterpolationMode.BICUBIC,
                                antialias=True).clamp(0,1)
                            avg_rgb = (imgs*mask_img).sum(2).sum(2)/mask_img[:, 0].sum()
                            # color_loss_weight usually 1.5
                            loss = torch.nn.functional.mse_loss(avg_rgb[0], torch.Tensor(rgb_val).cuda())*color_loss_weight*75
                            print(f'{object_name} loss: {loss}')
                            if loss.isnan():
                                if (not (loss_total >0)):
                                    flag=True
                            else:
                                # boost/lower color loss if color loss is too high/low
                                if loss > color_loss_weight*7 and self.diff_step>=40:
                                    loss = 4*loss
                                if loss < 1.5*color_loss_weight:
                                    loss = loss/4
                                flag=False
                                loss_total += loss
                        if not flag:
                            loss_total.backward()
                            latents_cur= (latents_cur - latents_cur.grad*total_mask).detach().clone()
                            print(f'loss_total step {self.diff_step}: {loss_total}')
                        else:
                            print(f"skipping color loss at step {self.diff_step}")
                    
                    if latents.shape[0]==3:
                        latents = torch.cat([latents_recon, latents_cur, latents_ref])
                    elif latents.shape[0]==2:
                        latents = torch.cat([latents_recon, latents_cur])
                    else:
                        latents = latents_cur

                    del imgs
                    del loss
                    del loss_total
                    torch.cuda.empty_cache()
                
                
            
            # object and background blending      
            if (self.diff_step>=background_blend_timestep_start) and (self.diff_step <= background_blend_timestep):
                if latents.shape[0]==3:
                    latents_uc, latents_cur, latents_ref = latents.chunk(3)
                    latents_for_masking = torch.cat([latents_uc, latents_uc, latents_uc])
                else: #assuming shape 2 latents.shape[0]==2:
                    latents_uc, latents_cur = latents.chunk(2)
                    latents_for_masking = torch.cat([latents_uc, latents_uc])
                
                latents = total_mask * latents + (1 - total_mask) * latents_for_masking
            
            # saves all self/cross attention maps for debug
            if debug:
                for idx in range(1,len(prompt[1:])+1):
                    show_cross_self_attention_maps(controller,
                                                   res={'cross':16,'self':32},
                                                   from_where=("up", "down"),
                                                   prompts=prompt, 
                                                   tokenizer=self.tokenizer,
                                                   select=idx,
                                                   out_dir=kwds['out_dir'],
                                                   diff_step=self.diff_step)
                
            self.diff_step+=1 
            if return_intermediates:
                latents_list.append(latents)
                pred_x0_list.append(pred_x0)

        image = self.latent2image(latents, return_type="pt")
        
        del latents
        torch.cuda.empty_cache()
        
        if return_intermediates:
            pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return image, pred_x0_list, latents_list
        return image, total_mask

