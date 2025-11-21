import sys
sys.path.append('.')
sys.path.append('enviroment/src/')
sys.path.append('enviroment/src/linguistic_binding_sd/')

from compute_loss import get_attention_map_index_to_wordpiece, split_indices, calculate_positive_loss, calculate_negative_loss, get_indices, start_token, end_token, align_wordpieces_indices, extract_attribution_indices, extract_attribution_indices_with_verbs, extract_attribution_indices_with_verb_root, extract_entities_only


def unify_lists(list_of_lists):
    def flatten(lst):
        for elem in lst:
            if isinstance(elem, list):
                yield from flatten(elem)
            else:
                yield elem

    def have_common_element(lst1, lst2):
        flat_list1 = set(flatten(lst1))
        flat_list2 = set(flatten(lst2))
        return not flat_list1.isdisjoint(flat_list2)

    lst = []
    for l in list_of_lists:
        lst += l
    changed = True
    while changed:
        changed = False
        merged_list = []
        while lst:
            first = lst.pop(0)
            was_merged = False
            for index, other in enumerate(lst):
                if have_common_element(first, other):
                    # If we merge, we should flatten the other list but not first
                    new_merged = first + [item for item in other if item not in first]
                    lst[index] = new_merged
                    changed = True
                    was_merged = True
                    break
            if not was_merged:
                merged_list.append(first)
        lst = merged_list

    return lst


def align_indices(tokenizer, prompt, spacy_pairs):
    wordpieces2indices = get_indices(tokenizer, prompt)
    paired_indices = []
    collected_spacy_indices = (
        set()
    )  # helps track recurring nouns across different relations (i.e., cases where there is more than one instance of the same word)

    for pair in spacy_pairs:
        curr_collected_wp_indices = (
            []
        )  # helps track which nouns and amods were added to the current pair (this is useful in sentences with repeating amod on the same relation (e.g., "a red red red bear"))
        for member in pair:
            for idx, wp in wordpieces2indices.items():
                if wp in [start_token, end_token]:
                    continue

                wp = wp.replace("</w>", "")
                if member.text.lower() == wp.lower():
                    if idx not in curr_collected_wp_indices and idx not in collected_spacy_indices:
                        curr_collected_wp_indices.append(idx)
                        break
                # take care of wordpieces that are split up
                elif member.text.lower().startswith(wp.lower()) and wp.lower() != member.text.lower():  # can maybe be while loop
                    wp_indices = align_wordpieces_indices(
                        wordpieces2indices, idx, member.text
                    )
                    # check if all wp_indices are not already in collected_spacy_indices
                    if wp_indices and (wp_indices not in curr_collected_wp_indices) and all(
                            [wp_idx not in collected_spacy_indices for wp_idx in wp_indices]):
                        curr_collected_wp_indices.append(wp_indices)
                        break

        for collected_idx in curr_collected_wp_indices:
            if isinstance(collected_idx, list):
                for idx in collected_idx:
                    collected_spacy_indices.add(idx)
            else:
                collected_spacy_indices.add(collected_idx)

        if curr_collected_wp_indices:
            paired_indices.append(curr_collected_wp_indices)
        else:
            print(f"No wordpieces were aligned for {pair} in _align_indices")

    return paired_indices 

def _extract_attribution_indices(tokenizer, prompt,doc):
    modifier_indices = []
    # extract standard attribution indices
    modifier_sets_1 = extract_attribution_indices(doc)
    modifier_indices_1 = align_indices(tokenizer, prompt, modifier_sets_1)
    if modifier_indices_1:
        modifier_indices.append(modifier_indices_1)

    # extract attribution indices with verbs in between
    modifier_sets_2 = extract_attribution_indices_with_verb_root(doc)
    modifier_indices_2 = align_indices(tokenizer, prompt, modifier_sets_2)
    if modifier_indices_2:
        modifier_indices.append(modifier_indices_2)

    modifier_sets_3 = extract_attribution_indices_with_verbs(doc)
    modifier_indices_3 = align_indices(tokenizer, prompt, modifier_sets_3)
    if modifier_indices_3:
        modifier_indices.append(modifier_indices_3)

    # make sure there are no duplicates
    modifier_indices = unify_lists(modifier_indices)
    print(f"Final modifier indices collected:{modifier_indices}")

    return modifier_indices


def extract_subject_indicies(prompt):
    import torch
    import base64
    import os
    import numpy as np
    from tqdm import tqdm
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    # # model_id = "stabilityai/stable-diffusion-2-1-base"
    model_id = 'CompVis/stable-diffusion-v1-4'
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    import spacy
    parser = spacy.load("en_core_web_trf")
    doc = parser(prompt)
    return _extract_attribution_indices(pipe.tokenizer, prompt,doc)

