"""Main script for differentially private synthesis of bioRxiv schema data via AIM."""

import argparse
import logging
import os
import ast
import json
import re
import pandas as pd
import domain
import generate

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def parse_schema_value(val):
    """Parse a schema JSON string into a dict."""
    if pd.isna(val):
        return {}
    val = str(val).strip()
    # Remove markdown code block markers if present
    if val.startswith('```'):
        val = re.sub(r'^```(?:json)?\s*', '', val)
        val = re.sub(r'\s*```$', '', val)
    val = val.strip()
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(val)
        except Exception:
            return {}

def _split_options(options_str):
  """Split options by '|' while respecting parentheses.

  For example, '*In Vitro* Assays (e.g.|ELISA|Western Blot)|Other'
  splits into ['*In Vitro* Assays (e.g.|ELISA|Western Blot)', 'Other'].
  """
  options = []
  current = ""
  paren_depth = 0
  for char in options_str:
    if char == '(':
      paren_depth += 1
      current += char
    elif char == ')':
      paren_depth -= 1
      current += char
    elif char == '|' and paren_depth == 0:
      if current.strip():
        options.append(current.strip())
      current = ""
    else:
      current += char
  if current.strip():
    options.append(current.strip())
  return options


def parse_schema_file(schema_path):
  """Parse a schema file into an attribute_domains dict.

  Schema files have lines like:
    "**attr_name**": "<opt1|opt2|opt3>", // comment

  Returns a dict mapping attribute names to domain.CategoricalAttribute.
  """
  with open(schema_path, 'r') as f:
    content = f.read()

  # Match "**attr_name**": "<options>" or "attr_name": "<options>"
  pattern = r'"(?:\*\*)?([^"*]+)(?:\*\*)?"\s*:\s*"<([^>]+)>"'
  matches = re.findall(pattern, content)

  attribute_domains = {}
  for attr_name, options_str in matches:
    options = _split_options(options_str)
    attribute_domains[attr_name] = domain.CategoricalAttribute(options)
    logging.info("  %s: %d categories", attr_name, len(options))

  return attribute_domains


def main():
  """Main function to load Biorxiv schema data, define the data domain,

  and generate synthetic data based on it.
  """

  parser = argparse.ArgumentParser(
      description="Generate synthetic Biorxiv schema data."
  )
  parser.add_argument("--eps", type=float, default=4.0)
  parser.add_argument(
      "--rho",
      type=float,
      default=0,
      help="Rho value for the synthetic data generation.",
  )
  parser.add_argument(
      "--pgm_iters",
      type=int,
      default=2000,
      help=(
          "Number of optimization iterations for the synthetic data generation."
      ),
  )
  parser.add_argument(
      "--num_gen",
      type=int,
      default=5000,
      help="Number of synthetic samples to generate.",
  )
  parser.add_argument(
      "--data_path",
      type=str,
      default="../data/biorxiv/biorxiv_schema_v2_train_gemini-2.5-flash_parsed.csv",
  )
  parser.add_argument(
      "--schema_path",
      type=str,
      default="../annotation/schema/biorxiv_schema_noexample_test.txt",
  )
  parser.add_argument(
      "--output_name",
      type=str,
      default="synthetic_biorxiv",
  )
  args = parser.parse_args()

  # --- 1. Load the real dataset ---
  data_path = os.path.join("../data", args.data_path)
  logging.info(f"Loading data from {data_path}...")
  try:
    raw_df = pd.read_csv(data_path)
  except FileNotFoundError:
    raise FileNotFoundError(f"Error: Data file not found at {data_path}")
  logging.info(f"Successfully loaded {len(raw_df)} records.")

  # Expand the 'schema' column JSON into separate columns, drop everything else
  df = pd.DataFrame(raw_df['schema'].apply(parse_schema_value).tolist())
  logging.info(f"Expanded schema into {len(df.columns)} attributes: {list(df.columns)}")

  # --- 2. Define the data schema (attribute_domains) ---
  logging.info("Defining data schema from %s...", args.schema_path)
  attribute_domains = parse_schema_file(args.schema_path)
  logging.info("Parsed %d attributes: %s", len(attribute_domains), list(attribute_domains.keys()))

  # --- 3. Run the synthetic data generation ---
  logging.info("Starting synthetic data generation...")
  t_start = pd.Timestamp.now()

  synth = generate.run(
      attribute_domains,
      df,
      epsilon=args.eps,
      delta=3.38e-6,
      rho=args.rho,
      num_records=args.num_gen,
      pgm_iters=args.pgm_iters,
  )
  logging.info(
      "Synthetic data generation completed in: %s", pd.Timestamp.now() - t_start
  )

  # --- 4. Save and inspect the output ---
  if args.rho:
    output_path = f"results/{args.output_name}_et_{args.num_gen//1000}k_rho-{args.rho}_iter-{args.pgm_iters}.csv"
  else:
    output_path = f"results/{args.output_name}_et_{args.num_gen//1000}k_eps-{args.eps}_iter-{args.pgm_iters}.csv"
  logging.info(f"Saving {len(synth)} generated records to {output_path}...")
  
  pd.DataFrame({"generated_text": synth.apply(lambda row: row.to_json(), axis=1)}).to_csv(output_path, index=False)

  logging.info(
      "Generation complete. Here are the first 5 rows of the synthetic data:"
  )
  logging.info("\n%s", synth.head())


if __name__ == "__main__":
  main()




# attribute_domains = {
#       "map_primary_research_area": domain.CategoricalAttribute([
#           "Biochemistry",
#           "Bioinformatics",
#           "Biophysics",
#           "Cancer Biology",
#           "Cell Biology",
#           "Clinical Trials",
#           "Developmental Biology",
#           "Ecology",
#           "Epidemiology",
#           "Evolutionary Biology",
#           "Genetics",
#           "Genomics",
#           "Immunology",
#           "Microbiology",
#           "Molecular Biology",
#           "Neuroscience",
#           "Paleontology",
#           "Pathology",
#           "Pharmacology and Toxicology",
#           "Physiology",
#           "Plant Biology",
#           "Public Health",
#           "Scientific Communication and Education",
#           "Structural Biology",
#           "Synthetic Biology",
#           "Systems Biology",
#           "Zoology",
#           "Other",
#       ]),
#       "map_model_organism": domain.CategoricalAttribute([
#           "Human",
#           "Mouse/Rat",
#           "Zebrafish",
#           "Drosophila melanogaster",
#           "Caenorhabditis elegans",
#           "Saccharomyces cerevisiae",
#           "Escherichia coli",
#           "Arabidopsis thaliana",
#           "Plant",
#           "Cell Culture",
#           "In Silico / Computational",
#           "Other Mammal",
#           "Other Vertebrate",
#           "Other Invertebrate",
#           "Other Microbe",
#           "Not Applicable / Review",
#           "Other",
#       ]),
#       "map_experimental_approach": domain.CategoricalAttribute([
#           "Wet Lab Experimentation",
#           "Computational / In Silico Analysis",
#           "Clinical Study",
#           "Field Study / Observation",
#           "Case Study / Case Review",
#           "Review / Meta-analysis",
#           "New Method Development",
#           "Theoretical Modeling",
#           "Other",
#       ]),
#       "map_dominant_data_type": domain.CategoricalAttribute([
#           "Genomic",
#           "Transcriptomic",
#           "Proteomic",
#           "Metabolomic",
#           "Imaging",
#           "Structural",
#           "Phenotypic / Behavioral",
#           "Ecological / Environmental",
#           "Clinical / Patient Data",
#           "Simulation / Model Output",
#           "Multi-omics",
#           "Other",
#       ]),
#       "map_research_focus_scale": domain.CategoricalAttribute([
#           "Molecular",
#           "Cellular",
#           "Circuit / Network",
#           "Tissue / Organ",
#           "Organismal",
#           "Population",
#           "Ecosystem",
#           "Multi-scale",
#           "Other",
#       ]),
#       "map_disease_mention": domain.CategoricalAttribute([
#           "Cancer",
#           "Neurodegenerative Disease",
#           "Infectious Disease",
#           "Metabolic Disease",
#           "Cardiovascular Disease",
#           "Autoimmune / Inflammatory Disease",
#           "Psychiatric / Neurological Disorder",
#           "Genetic Disorder",
#           "No Specific Disease Mentioned",
#           "Other",
#       ]),
#       "map_sample_size": domain.CategoricalAttribute([
#           "Single Subject / Case Study",
#           "Small Cohort (<50 subjects)",
#           "Medium Cohort (50-1000 subjects)",
#           "Large Cohort / Population-scale (>1000 subjects)",
#           "Relies on Cell/Animal Replicates",
#           "Not Specified / Not Applicable",
#       ]),
#       "map_research_goal": domain.CategoricalAttribute([
#           "Investigating a mechanism",
#           "Characterizing a system/molecule",
#           "Developing a method/tool",
#           "Identifying novel elements",
#           "Testing a hypothesis",
#           "Quantifying a parameter",
#           "Evaluating/Comparing approaches",
#           "Other",
#       ]),
#   }
