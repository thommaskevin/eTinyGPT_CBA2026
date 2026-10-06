from collections import defaultdict
import re

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"  # Start Of Sequence
EOS_TOKEN = "<eos>"  # End Of Sequence
UNK_TOKEN = "<unk>"  # Unknown Token
SEP_TOKEN = "<sep>"  # Separator between Question and Answer


class CustomTokenizer:
    """
    Custom tokenizer class that builds vocabulary from text or Q&A data,
    supports encoding words to token IDs and decoding token IDs back to text.
    It also manages special tokens with fixed IDs.
    """

    def __init__(self, data_source=None):
        """
        Initializes the tokenizer vocabulary. Adds special tokens first with fixed IDs.

        Args:
            data_source (str or list, optional): Text string or list of Q&A dicts 
                to build vocabulary from. If None, starts with only special tokens.
        """
        try:
            # defaultdict assigns new ID as current vocab size on first access
            self.word_to_idx = defaultdict(lambda: len(self.word_to_idx))
            self.idx_to_word = {}
            self.vocab_size = 0

            # Add special tokens with fixed low IDs first
            self.sos_id = self.word_to_idx[SOS_TOKEN]  # ID 0
            self.eos_id = self.word_to_idx[EOS_TOKEN]  # ID 1
            self.pad_id = self.word_to_idx[PAD_TOKEN]  # ID 2
            self.unk_id = self.word_to_idx[UNK_TOKEN]  # ID 3
            self.sep_id = self.word_to_idx[SEP_TOKEN]  # ID 4

            # Update reverse mapping
            self._update_idx_to_word()

            # Build vocabulary if data_source is provided
            if data_source:
                if isinstance(data_source, str):
                    self._build_vocab_from_text(data_source)
                elif isinstance(data_source, list):
                    # Assumes list of dicts with "question" and "answer" keys
                    self._build_vocab_from_qna_list(data_source)
                else:
                    raise ValueError(
                        "data_source for CustomTokenizer must be a string or a list of Q&A dicts."
                    )

            # Finalize vocabulary size
            self._update_idx_to_word()
            self.vocab_size = len(self.word_to_idx)

            # After vocab build, default to unknown token ID on unknown words
            self.word_to_idx.default_factory = lambda: self.unk_id

        except Exception as e:
            raise RuntimeError(f"Failed to initialize CustomTokenizer: {e}")

    def _update_idx_to_word(self):
        """
        Updates the reverse dictionary mapping from indices to words.
        """
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def _build_vocab_from_text(self, text_data):
        """
        Builds vocabulary from a multiline text string by tokenizing each line.

        Args:
            text_data (str): Raw text data.
        """
        try:
            for line in text_data.splitlines():
                tokens = self._tokenize_words(line)
                for token in tokens:
                    _ = self.word_to_idx[token]  # Access to add to defaultdict if missing
            self._update_idx_to_word()
            self.vocab_size = len(self.word_to_idx)
            print(f"Vocabulary built from text with {self.vocab_size} tokens.")
        except Exception as e:
            raise RuntimeError(f"Error building vocabulary from text: {e}")

    def _build_vocab_from_qna_list(self, qna_list):
        """
        Builds vocabulary from a list of Q&A dictionaries.

        Args:
            qna_list (list): List of dicts with keys "question" and "answer".
        """
        try:
            for item in qna_list:
                question_tokens = self._tokenize_words(item["question"])
                answer_tokens = self._tokenize_words(item["answer"])
                for token in question_tokens:
                    _ = self.word_to_idx[token]
                for token in answer_tokens:
                    _ = self.word_to_idx[token]
            self._update_idx_to_word()
            self.vocab_size = len(self.word_to_idx)
            print(f"[INFO] Vocabulary built from Q&A with {self.vocab_size} tokens.")
        except KeyError as e:
            raise KeyError(f"Missing expected key in Q&A data: {e}")
        except Exception as e:
            raise RuntimeError(f"Error building vocabulary from Q&A list: {e}")

    def _tokenize_words(self, text):
        """
        Tokenizes input text by removing punctuation and splitting on whitespace.

        Args:
            text (str): Input text string.

        Returns:
            list: List of lowercase tokens.
        """
        try:
            cleaned = text#re.sub(r'[^\w\s]', '', text).lower()  # Remove punctuation and lowercase
            return cleaned.split()
        except Exception as e:
            raise RuntimeError(f"Error tokenizing text '{text}': {e}")

    def encode_words(self, text):
        """
        Encodes input text into a list of token IDs.

        Args:
            text (str): Input text string.

        Returns:
            list: List of token IDs corresponding to words in the input.
        """
        try:
            tokens = self._tokenize_words(text)
            return [self.word_to_idx.get(token, self.unk_id) for token in tokens]
        except Exception as e:
            raise RuntimeError(f"Error encoding words: {e}")

    def decode_ids(self, token_ids):
        """
        Decodes a list of token IDs back into a text string, skipping special tokens.

        Args:
            token_ids (list): List of integer token IDs.

        Returns:
            str: Decoded text string.
        """
        try:
            words = []
            for token_id in token_ids:
                if token_id == self.pad_id:
                    continue  # Skip padding tokens
                elif token_id == self.sos_id:
                    continue  # Usually skip <sos> in output
                elif token_id == self.eos_id:
                    break     # Stop at end of sequence
                else:
                    words.append(self.idx_to_word.get(token_id, UNK_TOKEN))
            return " ".join(words)
        except Exception as e:
            raise RuntimeError(f"Error decoding token IDs: {e}")

    def load_vocab_dict(self, vocab_dict_loaded):
        """
        Loads vocabulary from an external dictionary or list while preserving special tokens.

        Args:
            vocab_dict_loaded (dict or list): Vocabulary mapping or list of tokens loaded from file.
        """
        try:
            # Preserve current special tokens with fixed IDs
            current_special_tokens = {
                SOS_TOKEN: self.sos_id,
                EOS_TOKEN: self.eos_id,
                PAD_TOKEN: self.pad_id,
                UNK_TOKEN: self.unk_id,
                SEP_TOKEN: self.sep_id
            }

            # Reset word_to_idx with fresh defaultdict
            self.word_to_idx = defaultdict(lambda: len(self.word_to_idx))

            # Add special tokens with fixed IDs
            for token, idx in current_special_tokens.items():
                self.word_to_idx[token] = idx

            # Handle vocab_dict_loaded differently if it's a list or dict
            if isinstance(vocab_dict_loaded, list):
                # Rebuild vocab from list, adding special tokens first
                self.sos_id = self.word_to_idx[SOS_TOKEN]
                self.eos_id = self.word_to_idx[EOS_TOKEN]
                self.pad_id = self.word_to_idx[PAD_TOKEN]
                self.unk_id = self.word_to_idx[UNK_TOKEN]
                self.sep_id = self.word_to_idx[SEP_TOKEN]

                for word in vocab_dict_loaded:
                    if word not in self.word_to_idx:
                        _ = self.word_to_idx[word]
            elif isinstance(vocab_dict_loaded, dict):
                # For dicts, more complex logic needed to avoid ID conflicts
                # Simplifying here by trusting loaded dict (could be improved)
                for word, idx in vocab_dict_loaded.items():
                    if word not in current_special_tokens:
                        self.word_to_idx[word] = idx
            else:
                raise ValueError("vocab_dict_loaded must be a list or dictionary.")

            # Update reverse mapping and vocab size
            self._update_idx_to_word()
            self.vocab_size = len(self.word_to_idx)

            # Ensure unknown token ID is valid and set as default factory
            self.word_to_idx.default_factory = lambda: self.unk_id
            if self.unk_id not in self.word_to_idx.values():
                print(f"Warning: UNK_TOKEN (ID: {self.unk_id}) not mapped correctly after loading vocab.")
                if UNK_TOKEN in self.word_to_idx:
                    self.unk_id = self.word_to_idx[UNK_TOKEN]
                else:
                    self.unk_id = self.word_to_idx[UNK_TOKEN]
                    self.vocab_size = len(self.word_to_idx)
                    self._update_idx_to_word()
                    print(f"UNK_TOKEN added to loaded vocabulary with ID {self.unk_id}.")

        except Exception as e:
            raise RuntimeError(f"Error loading vocabulary dictionary: {e}")

