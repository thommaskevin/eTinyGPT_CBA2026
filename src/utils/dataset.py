import json
import torch

from torch.utils.data import Dataset


class QADataset(Dataset):
    """
    Custom Dataset for Question-Answer pairs, preparing input and target sequences for training.

    Each sample is tokenized and formatted as:
    <SOS> question <SEP> answer <EOS>

    Inputs and targets are padded or truncated to `max_length`.
    """

    def __init__(self, qna_list = None, tokenizer = None, max_length = None):
        self.input_ids_list = []
        self.target_ids_list = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer != None: 
            sos_id = tokenizer.sos_id
            eos_id = tokenizer.eos_id
            sep_id = tokenizer.sep_id
            pad_id = tokenizer.pad_id
            max_len_found = 0

            for item in qna_list:
                # Encode question and answer separately
                q_tokens = tokenizer.encode_words(item["question"])
                a_tokens = tokenizer.encode_words(item["answer"])

                # Construct full sequence: <SOS> question <SEP> answer <EOS>
                full_sequence = [sos_id] + q_tokens + [sep_id] + a_tokens + [eos_id]
                max_len_found = max(max_len_found, len(full_sequence))

                # Prepare inputs (all tokens except last)
                input_seq = full_sequence[:-1]
                # Prepare targets (all tokens except first) for next-token prediction
                target_seq = full_sequence[1:]

                # Pad or truncate sequences to max_length
                input_seq_padded = input_seq[:max_length] + [pad_id] * max(0, max_length - len(input_seq))
                target_seq_padded = target_seq[:max_length] + [pad_id] * max(0, max_length - len(target_seq))

                # Store as torch tensors
                self.input_ids_list.append(torch.tensor(input_seq_padded, dtype=torch.long))
                self.target_ids_list.append(torch.tensor(target_seq_padded, dtype=torch.long))

            print(f"[INFO] Max original sequence length found: {max_len_found}")
            print(f"[INFO] Sequences adjusted to max_length: {max_length}")

    def __len__(self):
        """Returns the number of samples."""
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        """Returns a tuple of (input_ids, target_ids) tensors for the given index."""
        return self.input_ids_list[idx], self.target_ids_list[idx]


    def load_jsonl_file(self, filename):
        """
        Load a JSONL (JSON Lines) file and return a list of dictionaries.

        Args:
            filename (str): Path to the JSONL file.

        Returns:
            list: List of dictionaries parsed from each line of the file.
        """
        data = []
        try:
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    # Strip whitespace and skip empty lines
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))
            return data
        except FileNotFoundError:
            print(f"File '{filename}' not found.")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {line}\nError: {e}")
            return []