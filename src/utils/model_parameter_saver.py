from typing import Any
import logging

from src.utils.json_serializer import JsonSerializer
from src.utils.parameter_extractor import ParameterExtractor


class ModelParameterSaver:
    """
    Attributes:
        serializer (JsonSerializer): Handles writing dictionaries to JSON files.
        extractor (ParameterExtractor): Extracts parameter data from model.
    """

    def __init__(
        self,
        output_dir: str,
        logger: logging.Logger = None
    ) -> None:
        """
        Initializes the ModelParameterSaver.

        Args:
            output_dir (str): Directory where parameter JSON files will be saved.
            logger (logging.Logger, optional): Logger for status messages. 
                Defaults to module logger if None.
        """
        self.serializer = JsonSerializer(output_dir, logger)
        self.extractor = ParameterExtractor()

    def save_all(self, model: Any) -> None:
        """
        Extracts and saves all key parameter groups to JSON files.

        The following files are generated:
        - embedding.json
        - transformer_blocks.json
        - final_norm.json
        - out_head.json

        Args:
            model (Any): A model instance with properties expected by the extractor.

        Raises:
            IOError: If any extraction or saving operation fails.
        """
        try:
            # Save embedding parameters
            embeddings = self.extractor.extract_embeddings(model)
            self.serializer.save(embeddings, "embedding.json")

            # Save transformer block parameters
            trf_blocks = self.extractor.extract_transformer_blocks(model)
            self.serializer.save(trf_blocks, "transformer_blocks.json")

            # Save final normalization parameters
            final_norm = self.extractor.extract_final_norm(model)
            self.serializer.save(final_norm, "final_norm.json")

            # Save output head parameters
            out_head = self.extractor.extract_output_head(model)
            self.serializer.save(out_head, "out_head.json")

            # Log success message
            self.serializer.logger.info("All model parameters saved successfully.")

        except Exception as error:
            error_msg = f"Failed to save model parameters: {error}"
            # Log the error with stack trace
            self.serializer.logger.exception(error_msg)
            # Reraise as IOError for consistency
            raise IOError(error_msg) from error
