import argparse
import os
import re

import pandas as pd
from dotenv import load_dotenv
from google import genai


def parse_response_to_schema(response_text: str) -> str:
  # transform LLM output to schema json
  lines = response_text.strip().split('\n')

  features = []
  current_feature = {}

  for line in lines:
    line = line.strip()

    feature_match = re.match(r'^\d+\.\s*\*\*[^*]+\*\*:\s*(.+)', line)
    if feature_match:
      if current_feature:
        features.append(current_feature)
      current_feature = {'name': feature_match.group(1).strip()}
      continue

    values_match = re.match(r'^-?\s*\*\*Possible Values\*\*:\s*(.+)', line)
    if values_match:
      values_text = values_match.group(1).strip()
      values = re.findall(r'`([^`]+)`|"([^"]+)"', values_text)
      values = [v[0] or v[1] for v in values]
      if not values:
        values = [v.strip().strip('`"') for v in values_text.split(',')]
      current_feature['values'] = values
      continue

    desc_match = re.match(r'^-?\s*\*\*Description\*\*:\s*(.+)', line)
    if desc_match:
      current_feature['description'] = desc_match.group(1).strip()
      continue

  if current_feature:
    features.append(current_feature)

  schema_lines = []
  for i, feat in enumerate(features):
    name = feat.get('name', '').lower().replace(' ', '_')
    values = feat.get('values', [])
    desc = feat.get('description', '')

    values_str = '|'.join(values)

    comma = ',' if i < len(features) - 1 else ''
    schema_lines.append(f'  "{name}": "<{values_str}>"{comma} // {desc}')

  return '{{\n' + '\n'.join(schema_lines) + '\n}}'


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description='Design schema for abstracts using the Gemini API.'
  )
  parser.add_argument(
      '--output_name',
      type=str,
      required=True,
      help='Output file name (without .txt extension). Will be saved to annotation/schema/{output_name}.txt',
  )
  parser.add_argument(
      '--sample_size',
      type=int,
      default=5,
      help='Number of abstracts to sample as examples.',
  )
  args = parser.parse_args()

  load_dotenv()
  GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

  if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set.")

  client = genai.Client(api_key=GOOGLE_API_KEY)

  train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/biorxiv/train.csv"))
  sampled_abstracts = train_df['abstract'].dropna().sample(n=args.sample_size, random_state=42).tolist()
  examples = "\n\n".join([f"Example {i+1}:\n{abstract}" for i, abstract in enumerate(sampled_abstracts)])

  data_description = (
      "This is a dataset of biological research paper abstracts hosted on the"
      " bioRxiv preprint server."
  )
  workload_description = (
      "Help feature the category and main idea of the paper, for the purpose of"
      " e.g. assigning a reviewer."
  )
  num_features = 8

  prompt_template = open("prompts/schema_design_prompt_example.txt").read()
  input_prompt = prompt_template.format(
      data_description=data_description,
      workload_description=workload_description,
      num_features=num_features,
      examples=examples,
  )

  MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"
#   MODEL_NAME = "gemini-3-flash-preview"

  response = client.models.generate_content(
      model=MODEL_NAME,
      contents=input_prompt,
  )

  print(response.text.strip())

  schema_output = parse_response_to_schema(response.text)
  print("\n" + "="*50)
  print("Converted Schema:")
  print("="*50)
  print(schema_output)

  # save schema
  output_dir = os.path.join(os.path.dirname(__file__), "schema")
  os.makedirs(output_dir, exist_ok=True)
  output_path = os.path.join(output_dir, f"{args.output_name}.txt")

  with open(output_path, 'w', encoding='utf-8') as f:
    f.write(schema_output)

  print(f"\nSchema saved to: {output_path}")
