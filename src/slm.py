import math
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.gpt_model import GPTModel
from src.utils.code_generator import ArduinoCodeGenerator
from src.utils.dataset import QADataset
from src.utils.model_parameter_saver import ModelParameterSaver
from src.utils.tokenizer import CustomTokenizer


class TinyGPT:
    def __init__(self, embedding_dim: int, heads: int, layers: int, drop_rate: float):
        """
        Initializes the TinyGPT model configuration parameters.

        Args:
            embedding_dim (int): Dimension size of token embeddings.
            heads (int): Number of attention heads.
            layers (int): Number of transformer layers.
            drop_rate (float): Dropout rate for regularization.
        """
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.layers = layers
        self.drop_rate = drop_rate

        # Initialized later during training or loading
        self.model = None
        self.tokenizer = None
        self.config = None
        self.max_length = None


    def train(
        self,
        synthetic_dataset,
        num_epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.01,
        optimizer_cls=torch.optim.AdamW,
        loss_fn_cls=nn.CrossEntropyLoss
    ):
        """
        Trains the Q&A model on the given synthetic dataset.

        Args:
            synthetic_dataset (list or Dataset): Dataset to train on.
            num_epochs (int, optional): Number of epochs for training. Defaults to 1.
            batch_size (int, optional): Batch size. Defaults to 32.
            lr (float, optional): Learning rate. Defaults to 0.01.
            optimizer_cls (torch.optim.Optimizer class, optional): Optimizer class. Defaults to AdamW.
            loss_fn_cls (torch.nn.Module class, optional): Loss function class. Defaults to CrossEntropyLoss.

        Raises:
            ValueError: If the dataset is empty.
            Exception: Propagates exceptions occurring during training.
        """
        if not synthetic_dataset:
            raise ValueError("Dataset is empty.")

        try:
            # Initialize tokenizer and determine max sequence length
            self.tokenizer, self.max_length = self._init_tokenizer_and_max_length(synthetic_dataset)

            # Create DataLoader for batch training
            dataloader = self._create_dataloader(synthetic_dataset, self.tokenizer, batch_size, self.max_length)

            if len(dataloader.dataset) == 0:
                raise ValueError("Dataset is empty after preprocessing.")

            # Build model configuration dictionary
            self.config = self._build_model_config(self.tokenizer.vocab_size, self.max_length)

            # Select the appropriate device (GPU, MPS, or CPU)
            device = self._get_device()

            # Initialize model and move it to the selected device
            self.model = GPTModel(self.config).to(device)

            # Initialize optimizer and loss function
            optimizer = optimizer_cls(self.model.parameters(), lr=lr)
            loss_fn = loss_fn_cls(ignore_index=self.tokenizer.pad_id)

            # Train the model
            self._train_qna_model(self.model, dataloader, optimizer, loss_fn, device, num_epochs)

        except Exception as e:
            print(f"[ERROR] An error occurred during training: {e}")
            raise


    def _init_tokenizer_and_max_length(self, dataset, max_limit:int = 32, margin:int = 2):
        try:
            tokenizer = CustomTokenizer(dataset)
            max_len = max(
                1 + len(tokenizer.encode_words(item["question"])) +
                1 + len(tokenizer.encode_words(item["answer"])) + 1
                for item in dataset
            )
            max_len = min(max_len + margin, max_limit)
            print(f"[INFO] Context length: {max_len}")
            return tokenizer, max_len
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer or calculate max_length: {e}")

    def _create_dataloader(self, dataset, tokenizer, batch_size, max_length):
        """
        Creates a DataLoader for the Q&A dataset using the specified tokenizer and batch size.

        Args:
            dataset: Raw dataset to be wrapped by QADataset.
            tokenizer: Tokenizer instance for preprocessing.
            batch_size (int): Number of samples per batch.
            max_length (int): Maximum sequence length for tokenization.

        Returns:
            DataLoader: PyTorch DataLoader for the dataset.

        Raises:
            RuntimeError: If DataLoader creation fails.
        """
        try:
            # Wrap raw dataset with QADataset for preprocessing
            processed_dataset = QADataset(dataset, tokenizer, max_length)

            # Create DataLoader with shuffling enabled
            dataloader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            
            return dataloader

        except Exception as e:
            raise RuntimeError(f"Failed to create DataLoader: {e}")

        

    def _build_model_config(self, vocab_size, context_length, save_to_file=True):
        """
        Builds the model configuration dictionary based on input parameters and class attributes.

        Args:
            vocab_size (int): Size of the vocabulary.
            context_length (int): Maximum input sequence length.
            save_to_file (bool, optional): If True, saves the config to file (not implemented here).

        Returns:
            dict: Configuration dictionary for the model.
        """
        # Adjust heads to evenly divide embedding dimension
        heads = self._adjust_heads(self.embedding_dim, self.heads)

        config = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "emb_dim": self.embedding_dim,
            "n_heads": heads,
            "n_layers": self.layers,
            "drop_rate": self.drop_rate,
            "qkv_bias": False,
        }

        # Saving logic can be implemented here or handled externally
        if save_to_file:
            # Example: self._save_config(config)
            pass

        return config


    def _adjust_heads(self, emb_dim, heads):
        """
        Adjusts the number of attention heads to evenly divide the embedding dimension.

        Args:
            emb_dim (int): The embedding dimension size.
            heads (int): Desired number of attention heads.

        Returns:
            int: Adjusted number of heads that divides emb_dim evenly.

        Raises:
            ValueError: If no suitable number of heads can divide emb_dim evenly.
        """
        if emb_dim % heads == 0:
            return heads

        # Search for the largest divisor of emb_dim less than or equal to heads
        for candidate_heads in range(heads, 0, -1):
            if emb_dim % candidate_heads == 0:
                print(f"[INFO] Adjusted heads from {heads} to {candidate_heads} for embedding dim {emb_dim}.")
                return candidate_heads

        raise ValueError(f"No valid number of heads found that evenly divides embedding dimension {emb_dim}.")


    def _get_device(self):
        """
        Determines the best available device (CUDA, MPS, or CPU) for model execution.

        Returns:
            torch.device: The selected device.

        Logs:
            Prints which device is being used.
        """
        try:
            if torch.cuda.is_available():
                print("[INFO] Using CUDA (GPU).")
                return torch.device("cuda")

            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print("[INFO] Using Apple MPS (Metal Performance Shaders).")
                return torch.device("mps")

            else:
                print("[INFO] Using CPU.")
                return torch.device("cpu")

        except Exception as e:
            raise RuntimeError(f"Failed to determine device: {e}")


    def _train_qna_model(self, model, dataloader, optimizer, loss_fn, device, num_epochs):
        """
        Trains the Q&A model using the provided data and optimizer.

        Args:
            model: The model to be trained.
            dataloader: DataLoader providing input and target pairs.
            optimizer: Optimizer for model parameters.
            loss_fn: Loss function to compute training loss.
            device: Device to run training on (e.g., 'cpu' or 'cuda').
            num_epochs (int): Number of training epochs.

        Raises:
            RuntimeError: If any error occurs during training.
        """
        print("=" * 50)
        print("TRAINING MODEL".center(50))
        print("=" * 50)

        try:
            model.train()

            for epoch in range(num_epochs):
                total_loss = 0.0

                for input_ids, target_ids in dataloader:
                    # Move data to the target device
                    input_ids, target_ids = input_ids.to(device), target_ids.to(device)

                    # Reset gradients
                    optimizer.zero_grad()

                    # Forward pass
                    logits = model(input_ids)

                    # Compute loss
                    loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))

                    # Backward pass and optimizer step
                    loss.backward()
                    optimizer.step()

                    # Accumulate loss
                    total_loss += loss.detach().item()

                # Compute average loss for the epoch
                avg_loss = total_loss / len(dataloader)
                print(f"[INFO] Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

            model.eval()
            print("=" * 50)
            print("TRAINING COMPLETE".center(50))
            print("=" * 50)

        except Exception as e:
            raise RuntimeError(f"An error occurred during training: {e}")



    def save(self, output_folder: str = "output_folder"):
        """
        Saves the model, configuration, and tokenizer to the specified output folder.

        Args:
            output_folder (str): Destination directory for saving all artifacts.

        Raises:
            IOError: If any step in the saving process fails.
        """
        self.output_folder = output_folder

        try:
            # Save model configuration as JSON
            self._save_config(self.config)

            # Save model parameters (weights, biases, etc.)
            self._save_model_parameters(self.model)

            # Save tokenizer vocabulary list
            self._save_tokenizer_vocab(self.tokenizer)

            # Inform the user that the process was completed successfully
            print(f"[INFO] Model, configuration, and tokenizer successfully saved in '{self.output_folder}'")

        except (OSError, TypeError, ValueError) as e:
            # Raise a clear and specific error for known failure types
            raise IOError(f"Failed to save model/tokenizer due to data or I/O error: {e}")

        except Exception as e:
            # Handle any other unexpected exceptions
            raise IOError(f"Unexpected error while saving model/tokenizer: {e}")


    def _save_model_parameters(self, model):
        saver = ModelParameterSaver(output_dir=self.output_folder)
        saver.save_all(model)


    def _save_tokenizer_vocab(self, tokenizer):
        """
        Saves the tokenizer's vocabulary to a JSON file in the output folder.

        Args:
            tokenizer: A tokenizer instance with 'vocab_size' and 'word_to_idx' attributes.

        Raises:
            IOError: If saving the vocabulary fails due to file or serialization issues.
        """
        try:
            # Initialize a list to store vocabulary tokens
            vocab = [None] * tokenizer.vocab_size

            # Populate the vocabulary list with known tokens
            for word, idx in tokenizer.word_to_idx.items():
                if idx < tokenizer.vocab_size:
                    vocab[idx] = word
                else:
                    # Add extra tokens if index exceeds expected size
                    vocab.append(f"<extra_token_{idx}>")

            # Fill any missing entries with placeholder tokens
            vocab = [
                token if token is not None else f"<missing_{i}>"
                for i, token in enumerate(vocab)
            ]

            # Ensure the output directory exists
            os.makedirs(self.output_folder, exist_ok=True)

            # Define the path for the vocabulary file
            vocab_path = os.path.join(self.output_folder, "vocab_list.json")

            # Save the vocabulary list to a JSON file
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)

            # Log success message
            print(f"[INFO] Vocabulary successfully saved with {len(vocab)} tokens.")

        except (OSError, TypeError, ValueError) as e:
            # Raise a descriptive error if saving fails
            raise IOError(f"Failed to save tokenizer vocabulary: {e}")


    def _save_config(self, config):
        """
        Saves the model configuration dictionary as a JSON file in the output folder.

        Args:
            config (dict): Model configuration to be saved.

        Raises:
            IOError: If the configuration file cannot be saved.
        """
        try:
            # Ensure the output directory exists
            os.makedirs(self.output_folder, exist_ok=True)

            # Define the full path for the config file
            config_path = os.path.join(self.output_folder, "config.json")

            # Save the configuration dictionary as a JSON file
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            # Inform the user of successful save
            print(f"[INFO] Config successfully saved to: '{config_path}'")

        except (OSError, TypeError, ValueError) as e:
            # Raise a more descriptive error if saving fails
            raise IOError(f"Failed to save model config: {e}")

        

    def code_generator(self, model_folder, output_arduino_folder, model_name:str = 'gpt'):
        """
        Generates Arduino code from the exported JSON files of the model.

        Args:
            output_arduino_folder (str): Path to the folder where the Arduino code will be saved.
        """
        try:
            # List of required JSON files for code generation
            json_files = [
                os.path.join(model_folder, 'final_norm.json'),
                os.path.join(model_folder, 'vocab_list.json'),
                os.path.join(model_folder, 'transformer_blocks.json'),
                os.path.join(model_folder, 'embedding.json'),
                os.path.join(model_folder, 'out_head.json'),
                os.path.join(model_folder, 'config.json'),
            ]

            # Check if all JSON files exist
            for file in json_files:
                if not os.path.isfile(file):
                    raise FileNotFoundError(f"Missing JSON file: {file}")

            # Create an instance of the Arduino code generator
            generator = ArduinoCodeGenerator(output_dir=output_arduino_folder)

            # Generate Arduino code using the provided JSON files
            generator.generate_arduino_code(json_files, model_name)

            # Notify user where the generated files are located
            print(f"[INFO] Check the generated files at: {os.path.abspath(output_arduino_folder)}")

        except FileNotFoundError as fnf_error:
            print(f"File error: {fnf_error}")

        except Exception as e:
            print(f"Unexpected error during Arduino code generation: {e}")

    
    def evaluation(self, test_dataset, batch_size = 32, num_examples = 0):
        """
        Evaluates the model on a test dataset and computes various NLP metrics.
        
        Args:
            test_dataset: List of dicts {"question": ..., "answer": ...}
            batch_size: Batch size for evaluation
            num_examples: Number of examples to print for qualitative analysis
            
        Returns:
            dict: Dictionary containing evaluation metrics and sample outputs
            
        Raises:
            ValueError: If test_dataset is empty or invalid
            RuntimeError: If evaluation fails
        """
        try:
            # Input validation
            if not test_dataset or not isinstance(test_dataset, (list, tuple)):
                raise ValueError("test_dataset must be a non-empty list or tuple")
                
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")

            # 1. Prepare test dataloader
            test_dataloader = self._create_dataloader(
                test_dataset, 
                self.tokenizer, 
                batch_size, 
                self.max_length
            )
            
            # 2. Setup evaluation
            self.model.eval()
            device = self._get_device()
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)
            
            # Initialize metrics
            metrics = {
                'total_loss': 0.0,
                'total_items': 0,
                'correct_predictions': 0,
                'total_tokens': 0,
                'exact_matches': 0,
                'bleu_scores': [],
                'generated_examples': []
            }

            # 3. Calculate metrics on test dataset
            with torch.no_grad():
                for input_ids, target_ids in test_dataloader:
                    # Move data to device
                    input_ids = input_ids.to(device)
                    target_ids = target_ids.to(device)
                    
                    # Forward pass
                    logits = self.model(input_ids)
                    
                    # Calculate loss and perplexity
                    loss = loss_fn(
                        logits.view(-1, logits.size(-1)), 
                        target_ids.view(-1)
                    )
                    metrics['total_loss'] += loss.item()
                    metrics['total_items'] += 1
                    
                    # Calculate accuracy
                    preds = torch.argmax(logits, dim=-1)
                    metrics['correct_predictions'] += (preds == target_ids).sum().item()
                    metrics['total_tokens'] += (target_ids != self.tokenizer.pad_id).sum().item()
            
            # 4. Compute aggregate metrics
            avg_loss = metrics['total_loss'] / metrics['total_items']
            metrics.update({
                'avg_loss': avg_loss,
                'perplexity': math.exp(avg_loss),
                'token_accuracy': metrics['correct_predictions'] / metrics['total_tokens'],
                'num_samples': len(test_dataset)
            })
            
            # 5. Generate example outputs for qualitative analysis
            for i, example in enumerate(test_dataset[:num_examples]):
                try:
                    question = example["question"]
                    reference = example["answer"]
                    
                    # Encode and generate response
                    prompt_ids = [self.tokenizer.sos_id] + \
                                self.tokenizer.encode_words(question) + \
                                [self.tokenizer.sep_id]
                    
                    prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(device)
                    
                    generated_ids = self.generate_text_simple(
                        model=self.model,
                        idx_prompt=prompt_tensor,
                        max_new_tokens=50,
                        context_size=self.max_length,
                        tokenizer_obj=self.tokenizer
                    ).squeeze(0).tolist()
                    
                    # Decode and store results
                    generated_text = self.tokenizer.decode_ids(
                        generated_ids[len(prompt_ids):]
                    )
                    
                    # Calculate BLEU score
                    bleu = self._calculate_bleu(reference, generated_text)
                    
                    metrics['generated_examples'].append({
                        'question': question,
                        'reference': reference,
                        'generated': generated_text,
                        'bleu_score': bleu,
                        'exact_match': int(reference.lower() == generated_text.lower())
                    })
                    
                    metrics['bleu_scores'].append(bleu)
                    metrics['exact_matches'] += metrics['generated_examples'][-1]['exact_match']
                    
                except Exception as e:
                    print(f"Error processing example {i}: {str(e)}")
                    continue
            
            # Add aggregate text metrics
            if metrics['generated_examples']:
                metrics.update({
                    'avg_bleu': sum(metrics['bleu_scores']) / len(metrics['bleu_scores']),
                    'exact_match_rate': metrics['exact_matches'] / len(metrics['generated_examples'])
                })
            
            return metrics

        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}")


    def generate_text_simple(self, model, idx_prompt, max_new_tokens, context_size, tokenizer_obj):
        """
        Generates text from a prompt using greedy decoding.
        
        Args:
            model: The language model
            idx_prompt: Tensor (1, prompt_len) of initial token IDs
            max_new_tokens: Maximum number of tokens to generate
            context_size: Model's maximum context window size
            tokenizer_obj: Tokenizer instance for EOS detection
            
        Returns:
            Tensor: Generated sequence (prompt + generated tokens)
            
        Raises:
            RuntimeError: If generation fails
        """
        try:
            idx = idx_prompt.clone()  # Start with provided prompt
            
            for _ in range(max_new_tokens):
                # Truncate to context window if needed
                idx_cond = idx[:, -context_size:]
                
                # Get model predictions
                with torch.no_grad():
                    logits = model(idx_cond)
                
                # Get next token (greedy decoding)
                logits = logits[:, -1, :]  # Take last token's logits
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to sequence
                idx = torch.cat((idx, idx_next), dim=1)
                
                # Early stopping if EOS token generated
                if tokenizer_obj and (idx_next.item() == tokenizer_obj.eos_id):
                    break
                    
            return idx
        
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {str(e)}")


    def _calculate_bleu(self, reference, candidate):
        """
        Calculates BLEU score between reference and candidate text.
        
        Args:
            reference: Ground truth text
            candidate: Generated text
            
        Returns:
            float: BLEU-4 score
        """
        try:
            # Simple implementation - consider using NLTK's more complete version
            ref_tokens = reference.split()
            cand_tokens = candidate.split()
            
            if not ref_tokens or not cand_tokens:
                return 0.0
                
            # Calculate precision of n-grams (simplified)
            matches = sum(1 for token in cand_tokens if token in ref_tokens)
            precision = matches / len(cand_tokens)
            
            # Brevity penalty
            ratio = len(cand_tokens) / len(ref_tokens)
            bp = 1.0 if ratio >= 1.0 else math.exp(1 - 1/ratio)
            
            return bp * precision
            
        except Exception:
            return 0.0  # Return 0 if calculation fails
        
    def predict(self, input_question=''):  
        """  
        Generates a response to the input question using the language model.  
        
        Args:  
            input_question (str): The input question/prompt for which to generate a response.  
                                Defaults to an empty string.  
        
        Returns:  
            str: The model-generated response decoded into text.  
        
        Raises:  
            ValueError: If the input question is not a string.  
            RuntimeError: If an error occurs during text generation or decoding.  
        """  
        try:  
            # Input validation  
            if not isinstance(input_question, str):  
                raise ValueError("input_question must be a string")  
            
            if not input_question.strip():  
                return ""  # Return empty for blank or whitespace-only input  
                
            # Tokenize the input question  
            question_tokens = self.tokenizer.encode_words(input_question)  
            
            # Create prompt with special tokens (start of sentence and separator)  
            prompt_ids = [self.tokenizer.sos_id] + question_tokens + [self.tokenizer.sep_id]  
            
            # Convert to tensor and move to the appropriate device (CPU/GPU)  
            try:  
                prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(self._get_device())  
            except Exception as e:  
                raise RuntimeError(f"Failed to convert prompt to tensor: {str(e)}")  
            
            # Generate text sequence  
            try:  
                generated_sequence_tensor = self.generate_text_simple(  
                    model=self.model,  
                    idx_prompt=prompt_tensor,  
                    max_new_tokens=50,  
                    context_size=self.max_length,  
                    tokenizer_obj=self.tokenizer  
                ).squeeze(0).tolist()  
            except Exception as e:  
                raise RuntimeError(f"Text generation failed: {str(e)}")  
            
            # Extract only the answer part (excluding the prompt)  
            generated_answer_ids = generated_sequence_tensor[len(prompt_ids):]  
            
            # Decode answer tokens into text  
            try:  
                decoded_answer = self.tokenizer.decode_ids(generated_answer_ids)  
            except Exception as e:  
                raise RuntimeError(f"Failed to decode answer: {str(e)}")  
            
            return decoded_answer.strip()  # Remove leading/trailing whitespace  
        
        except Exception as e:  
            # Optional: Add logging here (e.g., self.logger.error(f"Predict error: {str(e)}"))  
            raise  # Re-raise the exception for the caller to handle  