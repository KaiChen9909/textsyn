import json
import numpy as np
import pandas as pd
import torch
import random
import logging
import os 

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logging.getLogger('google_genai.models').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

def eps_from_rho_subsample_exponential(rho, subsample_rate):
    """Convert rho to eps for subsampled Exponential mechanism.

    Args:
        rho: The rho value of the Exponential mechanism.
        subsample_rate: The Poisson subsampling rate.
    """
    total_eps = np.sqrt(8*rho)

    if subsample_rate == 1.0:
        return total_eps
    
    return np.log(1 + (np.exp(total_eps) - 1)/subsample_rate)


def exponential_mechanism(
    values: list[float], 
    eps: float, 
    sensitivity: float,
):  
    logging.info(f'Starting exponential mechanism, score list: \n{values}')
    logging.info(f'max score: {max(values)}, min score: {min(values)}')
    max_val = max(values)
    scores = [
        np.exp(eps * (val - max_val)/(2 * sensitivity))
        for val in values
    ]
    idx =  random.choices(range(len(scores)), weights=scores, k=1)[0]
    return idx 


def compute_next_token_loss(
    model,
    tokenizer,
    selected_attrs,
    df,
    prompt,
    text_column="abstract",
    batch_size=8,
    max_instruction_length=1024,
    max_answer_length=512,
    device=None,
    return_per_sample=True,
):
    """Compute next-token prediction loss for evaluating a candidate attribute.

    Constructs inputs from prompt + exist_attr values + candidate_attr value,
    with df[text_column] as the target output. Loss is computed only on the
    answer (text) tokens, not the instruction tokens.

    Args:
        model: A HuggingFace causal language model.
        tokenizer: The tokenizer corresponding to the model.
        selected_attrs: List of selected attribute names.
        df: DataFrame with columns [text_column, 'schema'].
            'schema' column contains JSON dict strings.
        prompt: Prompt template string with a {feature} placeholder,
            e.g. '<start_of_turn>user\n...{feature}\n<end_of_turn>\n<start_of_turn>model\n'
        text_column: Name of the text column (default: 'abstract').
        batch_size: Batch size for forward pass.
        max_instruction_length: Max token length for instruction part.
        max_answer_length: Max token length for answer part.
        device: Device to run on. If None, uses model's device.
        return_per_sample: If True, return per-sample average loss as a list.

    Returns:
        If return_per_sample=False: Average next-token prediction loss (float).
        If return_per_sample=True: (avg_loss, per_sample_losses) where
            per_sample_losses is a list of floats, one per row in df.
    """
    if device is None:
        device = next(model.parameters()).device

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build instructions and answers
    instructions = []
    answers = []
    for _, row in df.iterrows():
        schema = json.loads(row['schema']) if isinstance(row['schema'], str) else row['schema']
        sub_schema = {k: schema[k] for k in selected_attrs if k in schema}
        feature_str = json.dumps(sub_schema)
        instruction = prompt.format(feature=feature_str)
        instructions.append(instruction)
        answers.append(str(row[text_column]))

    ignore_index = -100
    total_loss = 0.0
    total_tokens = 0
    per_sample_losses = [] if return_per_sample else None

    model.eval()
    with torch.no_grad():
        for i in range(0, len(instructions), batch_size):
            batch_instructions = instructions[i:i + batch_size]
            batch_answers = answers[i:i + batch_size]

            tokenized_instructions = tokenizer(
                batch_instructions,
                max_length=max_instruction_length,
                padding=False,
                truncation=True,
            )
            tokenized_answers = tokenizer(
                batch_answers,
                max_length=max_answer_length,
                padding=False,
                truncation=True,
                add_special_tokens=False,
            )

            all_input_ids = []
            all_labels = []
            sample_answer_lengths = []  # track answer length per sample
            for j in range(len(batch_instructions)):
                instr_ids = tokenized_instructions['input_ids'][j]
                ans_ids = tokenized_answers['input_ids'][j]
                input_ids = instr_ids + ans_ids + [tokenizer.eos_token_id]
                labels = (
                    [ignore_index] * len(instr_ids)
                    + ans_ids
                    + [tokenizer.eos_token_id]
                )
                all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                all_labels.append(torch.tensor(labels, dtype=torch.long))
                sample_answer_lengths.append(len(ans_ids) + 1)  # +1 for eos

            # Pad to same length within batch
            input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                all_input_ids, batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ).to(device)
            labels_padded = torch.nn.utils.rnn.pad_sequence(
                all_labels, batch_first=True,
                padding_value=ignore_index,
            ).to(device)
            attention_mask = input_ids_padded.ne(tokenizer.pad_token_id).to(device)

            if return_per_sample:
                # Compute per-sample loss manually
                outputs = model(
                    input_ids=input_ids_padded,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits  # (batch, seq_len, vocab)
                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels_padded[:, 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
                # (batch, seq_len-1)
                loss_per_token = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_labels.size())

                # Sum per sample, then average by answer token count
                for j in range(loss_per_token.size(0)):
                    sample_loss = loss_per_token[j].sum().item()
                    sample_tokens = sample_answer_lengths[j]
                    per_sample_losses.append(sample_loss / sample_tokens if sample_tokens > 0 else 0.0)
                    total_loss += sample_loss
                    total_tokens += sample_tokens
            else:
                outputs = model(
                    input_ids=input_ids_padded,
                    attention_mask=attention_mask,
                    labels=labels_padded,
                )
                # outputs.loss is averaged over non-ignored tokens
                num_answer_tokens = (labels_padded != ignore_index).sum().item()
                total_loss += outputs.loss.item() * num_answer_tokens
                total_tokens += num_answer_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    if return_per_sample:
        return avg_loss, per_sample_losses
    return avg_loss


def filter_and_save_selected_attrs(
    selected_attrs,
    dataset_name,
    from_attr_num,
    select_attr_num,
    data_dir="../data",
    schema_dir="schema",
):
    """Filter schema and data files to only include selected attributes.

    Args:
        selected_attrs: List of selected attribute names.
        dataset_name: Name of the dataset (e.g., 'biorxiv').
        from_attr_num: Original number of attributes (e.g., 24).
        select_attr_num: Number of attributes selected (e.g., 8).
        data_dir: Directory containing data files.
        schema_dir: Directory containing schema files.

    Returns:
        Tuple of (new_schema_path, new_train_path, new_valid_path).
    """
    # Read and filter schema file
    original_schema_path = os.path.join(
        schema_dir, f"{dataset_name}_schema_noexample_{from_attr_num}attr.txt"
    )
    new_schema_path = os.path.join(
        schema_dir, f"{dataset_name}_schema_noexample_{select_attr_num}outof{from_attr_num}attr.txt"
    )

    logging.info(f"Reading original schema from {original_schema_path}")
    with open(original_schema_path, "r") as f:
        schema_content = f.read()

    # Parse the schema JSON-like structure
    schema_lines = schema_content.strip().split("\n")
    filtered_lines = [schema_lines[0]]  # Keep opening brace "{"

    for line in schema_lines[1:-1]: 
        # Check if this line contains any of the selected attributes
        for attr in selected_attrs: 
            if f'"**{attr}**"' in line:
                filtered_lines.append(line)
                break

    # Fix trailing commas: remove comma from last attribute line
    if len(filtered_lines) > 1 and filtered_lines[-1].rstrip().endswith(","):
        filtered_lines[-1] = filtered_lines[-1].rstrip().rstrip(",")

    filtered_lines.append(schema_lines[-1])

    # Save filtered schema
    os.makedirs(os.path.dirname(new_schema_path) if os.path.dirname(new_schema_path) else ".", exist_ok=True)
    with open(new_schema_path, "w") as f:
        f.write("\n".join(filtered_lines))
    logging.info(f"Saved filtered schema to {new_schema_path}")


    # Read and filter data files (train and valid)
    def filter_data_file(input_path, output_path):
        logging.info(f"Reading data from {input_path}")
        df = pd.read_csv(input_path)

        # Filter schema column to only include selected attributes
        new_schemas = []
        for schema_str in df["schema"]:
            schema = json.loads(schema_str) if isinstance(schema_str, str) else schema_str
            filtered_schema = {k: schema[k] for k in selected_attrs if k in schema}
            new_schemas.append(json.dumps(filtered_schema))

        df["schema"] = new_schemas

        # Save filtered data
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Saved filtered data ({len(df)} rows) to {output_path}")
        return output_path

    # Process train and valid files
    train_input = os.path.join(
        data_dir, dataset_name, f"clean_{dataset_name}_schema_noexample_{from_attr_num}attr_train.csv"
    )
    train_output = os.path.join(
        data_dir, dataset_name, f"clean_{dataset_name}_schema_noexample_{select_attr_num}outof{from_attr_num}attr_train.csv"
    )
    valid_input = os.path.join(
        data_dir, dataset_name, f"clean_{dataset_name}_schema_noexample_{from_attr_num}attr_valid.csv"
    )
    valid_output = os.path.join(
        data_dir, dataset_name, f"clean_{dataset_name}_schema_noexample_{select_attr_num}outof{from_attr_num}attr_valid.csv"
    )

    new_train_path = filter_data_file(train_input, train_output)
    new_valid_path = filter_data_file(valid_input, valid_output)

    return new_schema_path, new_train_path, new_valid_path


def iterative_attribute_selection(
    df,
    model_name_or_path,
    device,
    prompt,
    select_attr_num,
    text_column,
    rho,
    sample_rate=0.02,
):
    all_attrs = list(json.loads(df['schema'].iloc[0]).keys())
    all_selected_attrs = []
    candidate_attrs=[attr for attr in all_attrs if attr not in all_selected_attrs]
    eps_single_round = eps_from_rho_subsample_exponential(rho/select_attr_num, subsample_rate=sample_rate)

    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        attn_implementation='eager',
    )
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    for _ in tqdm(range(select_attr_num)):
        sample_mask = np.random.binomial(n=1, p=sample_rate, size=len(df)).astype(bool)
        df_sample = df[sample_mask].reset_index(drop=True)

        _, prev_loss = compute_next_token_loss(
            model,
            tokenizer,
            all_selected_attrs,
            df_sample,
            prompt,
            text_column=text_column,
            batch_size=8,
            max_instruction_length=1024,
            max_answer_length=512,
            device=device,
        )
        NLC = []

        for attr in candidate_attrs:
            temp_selected_attrs = all_selected_attrs + [attr]
            _, after_loss = compute_next_token_loss(
                model,
                tokenizer,
                temp_selected_attrs,
                df_sample,
                prompt,
                text_column=text_column,
                batch_size=8,
                max_instruction_length=1024,
                max_answer_length=512,
                device=device,
            )
            loss_change = [(prev_loss[i]-after_loss[i])/(prev_loss[i]+after_loss[i]) for i in range(len(prev_loss))]
            NLC.append(sum(loss_change))
        
        selected_idx = exponential_mechanism(
            NLC,
            eps=eps_single_round,
            sensitivity=2.0,
        )
        all_selected_attrs.append(candidate_attrs[selected_idx])
        logging.info(f'Selected attribute: {candidate_attrs[selected_idx]}')
        candidate_attrs=[attr for attr in all_attrs if attr not in all_selected_attrs] # update candidate attrs

    return all_selected_attrs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Iterative attribute selection for schema-based text generation")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Path to the CSV file containing the dataset (must have 'schema' and text columns)",
    )
    parser.add_argument(
        "--select_attr_num",
        type=int,
        required=True,
        help="Number of attributes to select",
    )
    parser.add_argument(
        "--rho",
        type=float,
        required=True,
        help="Privacy budget (rho) for the exponential mechanism",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/gemma-3-1b-pt",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="abstract",
        help="Name of the text column in the dataset (default: abstract)",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.02,
        help="Poisson subsampling rate (default: 0.02)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the selected attributes (optional, prints to stdout if not specified)",
    )
    parser.add_argument(
        "--from_attr_num",
        type=int,
        default=24,
        help="Number of attributes to start with (default: 24)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    data_path = f"../data/{args.dataset_name}/clean_{args.dataset_name}_schema_noexample_{args.from_attr_num}attr_train.csv"
    logging.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logging.info(f"Loaded {len(df)} samples")

    # Build prompt (biorxiv_noexample style)
    instruction = (
        "Please generate a synthetic scientific abstract that belongs to the"
        " below category, in the style of a bioRxiv paper."
    )
    prompt = f"<start_of_turn>user\n{instruction}\n\n{{feature}}\n<end_of_turn>\n<start_of_turn>model\n"

    # Run iterative attribute selection
    logging.info("Starting iterative attribute selection...")
    selected_attrs = iterative_attribute_selection(
        df=df,
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        prompt=prompt,
        select_attr_num=args.select_attr_num,
        text_column=args.text_column,
        rho=args.rho,
        sample_rate=args.sample_rate,
    )

    logging.info(f"Selected attributes: {selected_attrs}")

    # Save or print results
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(selected_attrs, f, indent=2)
        logging.info(f"Saved selected attributes to {args.output_path}")
    else:
        print("Selected attributes:")
        for i, attr in enumerate(selected_attrs, 1):
            print(f"  {i}. {attr}")
    
    # Update schema file and data files to only include selected attributes
    new_schema_path, new_train_path, new_valid_path = filter_and_save_selected_attrs(
        selected_attrs=selected_attrs,
        dataset_name=args.dataset_name,
        from_attr_num=args.from_attr_num,
        select_attr_num=args.select_attr_num,
        data_dir="../data",
        schema_dir="schema",
    )
    logging.info(f"Created new schema file: {new_schema_path}")
    logging.info(f"Created new train data file: {new_train_path}")
    logging.info(f"Created new valid data file: {new_valid_path}")


if __name__ == "__main__":
    main()
