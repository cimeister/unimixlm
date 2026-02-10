"""
Environment variable setup for the multilingual tokenization experiment.
This module sets up environment variables needed for the experiment.
"""

import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment():
    """
    Set up environment variables for the experiment.
    """
    # Set TOKENIZERS_PARALLELISM to true to avoid warnings from HuggingFace tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    logger.info(
        "Set TOKENIZERS_PARALLELISM=true to enable parallel processing in HuggingFace tokenizers"
    )

    # Add any other environment variables needed for the experiment here

    return True
