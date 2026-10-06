class ParameterExtractor:
    """
    Extracts parameters from model components into a serializable dictionary format.
    This class assumes the model has certain attributes like token embeddings, 
    positional embeddings, transformer blocks, final normalization layers, and output head.
    """

    @staticmethod
    def to_list(tensor):
        """
        Converts a PyTorch tensor to a nested Python list after detaching it from the graph 
        and moving it to CPU. Returns None if the tensor is None.
        
        Args:
            tensor (torch.Tensor or None): The tensor to convert.

        Returns:
            list or None: The converted list or None if input was None.
        """
        try:
            if tensor is None:
                return None
            # Detach tensor, move to CPU and convert to list
            return tensor.detach().cpu().numpy().tolist()
        except Exception as e:
            # Log or raise exception with context
            raise RuntimeError(f"Failed to convert tensor to list: {e}")

    def extract_embeddings(self, model):
        """
        Extracts token and positional embeddings from the model.

        Args:
            model: The model instance containing embeddings.

        Returns:
            dict: A dictionary with embeddings serialized as lists.
        """
        try:
            return {
                "tok_emb.weight": self.to_list(getattr(model.tok_emb, "weight", None)),
                "pos_emb.weight": self.to_list(getattr(model.pos_emb, "weight", None))
            }
        except AttributeError as e:
            raise AttributeError(f"Model is missing expected embedding attributes: {e}")

    def extract_transformer_blocks(self, model):
        """
        Extracts all transformer blocks parameters from the model.

        Args:
            model: The model instance containing transformer blocks.

        Returns:
            dict: A dictionary where each key corresponds to a block and its parameters.
        """
        blocks = {}
        try:
            for i, block in enumerate(getattr(model, "trf_blocks", [])):
                blocks[f"block_{i}"] = {
                    "att.W_query.weight": self.to_list(getattr(block.att.W_query, "weight", None)),
                    "att.W_query.bias": self.to_list(getattr(block.att.W_query, "bias", None)),
                    "att.W_key.weight": self.to_list(getattr(block.att.W_key, "weight", None)),
                    "att.W_key.bias": self.to_list(getattr(block.att.W_key, "bias", None)),
                    "att.W_value.weight": self.to_list(getattr(block.att.W_value, "weight", None)),
                    "att.W_value.bias": self.to_list(getattr(block.att.W_value, "bias", None)),
                    "att.out_proj.weight": self.to_list(getattr(block.att.out_proj, "weight", None)),
                    "att.out_proj.bias": self.to_list(getattr(block.att.out_proj, "bias", None)),
                    "norm1.scale": self.to_list(getattr(block.norm1, "scale", None)),
                    "norm1.shift": self.to_list(getattr(block.norm1, "shift", None)),
                    "norm2.scale": self.to_list(getattr(block.norm2, "scale", None)),
                    "norm2.shift": self.to_list(getattr(block.norm2, "shift", None)),
                    "ff.lin1.weight": self.to_list(getattr(block.ff.layers[0], "weight", None)),
                    "ff.lin1.bias": self.to_list(getattr(block.ff.layers[0], "bias", None)),
                    "ff.lin2.weight": self.to_list(getattr(block.ff.layers[2], "weight", None)),
                    "ff.lin2.bias": self.to_list(getattr(block.ff.layers[2], "bias", None)),
                }
            return blocks
        except AttributeError as e:
            raise AttributeError(f"Model's transformer blocks are missing expected attributes: {e}")
        except IndexError as e:
            raise IndexError(f"Unexpected structure in feed-forward layers: {e}")

    def extract_final_norm(self, model):
        """
        Extracts the final normalization layer parameters from the model.

        Args:
            model: The model instance containing final normalization.

        Returns:
            dict: A dictionary with scale and shift parameters serialized as lists.
        """
        try:
            return {
                "final_norm.scale": self.to_list(getattr(model.final_norm, "scale", None)),
                "final_norm.shift": self.to_list(getattr(model.final_norm, "shift", None))
            }
        except AttributeError as e:
            raise AttributeError(f"Model is missing final normalization attributes: {e}")

    def extract_output_head(self, model):
        """
        Extracts the output head weights from the model.

        Args:
            model: The model instance containing the output head.

        Returns:
            dict: A dictionary with the output head weights serialized as lists.
        """
        try:
            return {
                "out_head.weight": self.to_list(getattr(model.out_head, "weight", None))
            }
        except AttributeError as e:
            raise AttributeError(f"Model is missing output head attributes: {e}")
