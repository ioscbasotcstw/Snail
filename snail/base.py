from typing import List, Tuple, Dict
from abc import ABC, abstractmethod


class BaseDatasetGenerator(ABC):

    @abstractmethod
    def searching(self) -> str:
        """
        Generate content based on the user query using the Google Search tool.

        Returns:
            str: The generated text response from the GenAI model.
        """
        pass

    @abstractmethod
    def get_result(self, data: List[str], delay: int) -> List[str]:
        """
        Process a list of data entries using the Chain of Thought (CoT) framework and return results.

        Args:
            data (List[str]): List of data entries (e.g., statements or quotes) to analyze.
            delay (int): Time in seconds to wait between processing each entry to avoid rate limits.

        Returns:
            List[str]: List of generated text responses for each data entry.
        """
        pass

    @abstractmethod
    def create_ds(self, instruction: List[str], output: List[str]) -> Dict[str, str]:
        """
        Create a dictionary by combining instruction and output lists.

        Args:
            instruction (List[str]): List of instruction strings.
            output (List[str]): List of corresponding output strings.

        Returns:
            Dict[str, str]: A dictionary mapping instructions to their outputs.
        """
        pass

    @abstractmethod
    def transform_alpaca_format(self, dataset: Dict[str, str]) -> Tuple[str, List[Dict[str, str]]]:
        """
        Transform a dictionary dataset into Alpaca format and save to a JSON file.

        Args:
            dataset (Dict[str, str]): A dictionary of instructions and outputs.

        Returns:
            Tuple[str, List[Dict]]:
            - The filename of the saved JSON file
            - The list of transformed data entries
        """
        pass

    @abstractmethod
    def push_to_hf(self, json_path: str, repo_id: str) -> None:
        """
        Push a JSON dataset and its dataset card to a new Hugging Face dataset repository.

        Args:
            json_path (str): File path to the JSON file containing the dataset.
            repo_id (str): Repository identifier in the format 'username/reponame'.
        """
        pass