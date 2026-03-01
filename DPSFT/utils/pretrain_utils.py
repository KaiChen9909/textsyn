import json
import logging
import os
import os.path as osp

from datasets import Dataset, DatasetDict
from datasets import load_dataset, load_from_disk


def get_pretrain_generation_prompt_dict(prompt_template, prompt_type):
  if prompt_type == 'gen':
    if prompt_template.startswith('biorxiv_noexample') or prompt_template.startswith('biorxiv_condgen'):
      prompt_type = '_'.join(prompt_template.split('_')[:-1])
      instruction = (
          'Please generate a synthetic scientific abstract (ONLY abstract text) that belongs to the'
          ' JSON summary, in the style of a bioRxiv paper.'
          # ' Provide the abstract text only, starting directly with the content without any headers, labels, or introductory text.'
      )
      PROMPT_DICT = {
          'type': prompt_type,
          'prompt': '<start_of_turn>user\n{instruction}\n\n{{feature}}\n<end_of_turn>\n<start_of_turn>model\n'.format(
              instruction=instruction
          ),
      }
    else:
        raise ValueError(f"Unsupported prompt template: {prompt_template}")
    
  elif prompt_type == 'variation':
    if prompt_template.startswith('biorxiv_noexample') or prompt_template.startswith('biorxiv_condgen'):
      prompt_type = '_'.join(prompt_template.split('_')[:-1])
      instruction = (
          'Please rephrase the given synthetic scientific abstract (ONLY abstract text), ensuring it belongs to the'
          ' JSON summary, in the style of a bioRxiv paper.'
      )
      PROMPT_DICT = {
          'type': prompt_type,
          'prompt': '<start_of_turn>user\n{instruction}\n\nOriginal abstract: {{text}}\n\nJSON summary: {{feature}}\n<end_of_turn>\n<start_of_turn>model\n'.format(
              instruction=instruction
          ),
      }
    else:
        raise ValueError(f"Unsupported prompt template: {prompt_template}")
  else:
    raise ValueError(f"Unsupported prompt type: {prompt_type}. Must be 'gen' or 'variation'")

  return PROMPT_DICT



def get_pretrain_generation_prompt_dict_noschema(prompt_template, prompt_type):
  """Same as get_pretrain_generation_prompt_dict but without JSON schema (feature) in the prompt."""
  if prompt_type == 'gen':
    if prompt_template.startswith('biorxiv_noexample') or prompt_template.startswith('biorxiv_condgen') or prompt_template.startswith('biorxiv_noft'):
      instruction = (
          'Please generate a synthetic scientific abstract (ONLY abstract text, NO title, label, or introductory)'
          ' in the style of a bioRxiv paper.'
      )
      PROMPT_DICT = {
          'type': prompt_template,
          'prompt': instruction,
      }
    else:
      raise ValueError(f"Unsupported prompt template: {prompt_template}")

  elif prompt_type == 'variation':
    if prompt_template.startswith('biorxiv_noexample') or prompt_template.startswith('biorxiv_condgen') or prompt_template.startswith('biorxiv_noft'):
      instruction = (
          'Please rephrase the given synthetic scientific abstract (ONLY abstract text, NO title, label, or introductory)'
          ' in the style of a bioRxiv paper.'
      )
      PROMPT_DICT = {
          'type': prompt_template,
          'prompt': '{instruction}\n\nOriginal abstract: {{text}}'.format(instruction=instruction),
      }
    else:
      raise ValueError(f"Unsupported prompt template: {prompt_template}")
  else:
    raise ValueError(f"Unsupported prompt type: {prompt_type}. Must be 'gen' or 'variation'")

  return PROMPT_DICT


# def get_pretrain_generation_prompt_dict(prompt_template, prompt_type):
#   if prompt_type == 'gen':
#     if prompt_template.startswith('biorxiv_noexample') or prompt_template.startswith('biorxiv_condgen'):
#       prompt_type = '_'.join(prompt_template.split('_')[:-1])
#       instruction = (
#             "Please generate a synthetic scientific abstract based on the provided JSON summary in the style of a bioRxiv paper.\n"
#             "Requirement: Output ONLY the abstract content. Do NOT include titles, labels, headers, or introductory text."
#         )
#       PROMPT_DICT = {
#           'type': prompt_type,
#           'prompt': '<start_of_turn>user\n{instruction}\n\nJSON Summary:\n{{feature}}\n<end_of_turn>\n<start_of_turn>model\n'.format(
#               instruction=instruction
#           ),
#       }
#     else:
#         raise ValueError(f"Unsupported prompt template: {prompt_template}")
    
#   elif prompt_type == 'variation':
#     if prompt_template.startswith('biorxiv_noexample') or prompt_template.startswith('biorxiv_condgen'):
#       prompt_type = '_'.join(prompt_template.split('_')[:-1])
#       instruction = (
#             "Please rephrase the given scientific abstract in the style of a bioRxiv paper.\n"
#             "Requirements:\n"
#             "1. Ensure the rephrased text remains semantically consistent with the provided JSON summary.\n"
#             "2. Output ONLY the rephrased abstract content. Do NOT include titles, labels, headers, or introductory text."
#         )
#       PROMPT_DICT = {
#           'type': prompt_type,
#           'prompt': '<start_of_turn>user\n{instruction}\n\nOriginal Abstract:\n{{text}}\n\nJSON Summary:\n{{feature}}\n<end_of_turn>\n<start_of_turn>model\n'.format(
#               instruction=instruction
#           ),
#       }
#     else:
#         raise ValueError(f"Unsupported prompt template: {prompt_template}")
#   else:
#     raise ValueError(f"Unsupported prompt type: {prompt_type}. Must be 'gen' or 'variation'")

#   return PROMPT_DICT