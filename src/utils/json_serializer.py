import os
import json
from typing import Any, Dict, Optional
import logging

class JsonSerializer:
    """
    Handles saving dictionaries to JSON files.

    Attributes:
        output_dir (str): Directory where JSON files will be saved.
        logger (logging.Logger): Optional logger for structured logging.
    """

    def __init__(self, output_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initializes the JsonSerializer.

        Args:
            output_dir (str): Path to the directory for saving JSON files.
            logger (logging.Logger, optional): Logger instance for messages.
                                                Defaults to the root logger.
        """
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, data: Dict[str, Any], filename: str) -> None:
        """
        Saves a dictionary as a JSON file.

        Args:
            data (Dict[str, Any]): The data to serialize.
            filename (str): The name of the JSON file (e.g., "config.json").

        Raises:
            IOError: If the file cannot be written or data is not serializable.
        """
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved JSON to '{path}'.")
        except (OSError, TypeError, ValueError) as e:
            error_msg = f"Failed to save '{filename}': {e}"
            self.logger.error(error_msg)
            raise IOError(error_msg)

