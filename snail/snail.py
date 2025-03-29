import re
import time
import json
from datetime import datetime
from typing import List, Dict, Tuple
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from datasets import load_dataset
from huggingface_hub import create_repo, DatasetCard

DATASET_CARD = \
"""---
license: apache-2.0
language:
- en
task_categories:
- text-generation
size_categories:
- n<1K
---

# Uploaded dataset

- **Developed by:** {username}
- **License:** apache-2.0

# Made with Snail

[<img src="https://raw.githubusercontent.com/ioscbasotcstw/Snail/main/snail_mascot.png" width="200"/>](https://github.com/ioscbasotcstw/Snail)
"""


class Snail:
    """A class for generating CoT dataset using Google's GenAI API with search and Chain of Thought (CoT) capabilities."""

    def __init__(self,
                 google_api_key: str,
                 model_id: str,
                 role: str,
                 user_query: str,
                 max_output_tokens: int = 2048) -> None:
        """
        Initialise the Snail class with the necessary parameters to generate CoT ds.
        Very important for the step where you will push your dataset to HF,
        If you are using Colab, set up `!huggingface-cli login` and pass your token to HF.
        I recommend running this on Colab because it's easier and much more flexible.

        Example of usage a tool:
            >>> from snail import Snail

            >>> snail = Snail(google_api_key=google_api, model_id='gemini-2.0-flash-thinking-exp-01-21', user_query="List a 20 math problems from easiest to hard and numerate their", role="mathematician")

            >>> result = snail.searching()

            >>> instruction = snail.extract_listings(result)

            >>> # Modified these instructions at your task

            >>> snail.system_instruction_cot = f'''You are a {snail.role} expert skilled at explaining step by step mathematician problems, using a Chain of Thought (CoT) framework. Your response must include:
            - A thought process inside <thought></thought> tags, where you analyze the problem.
            - A final response inside <answer></answer> tags, solving the problem.
            Ensure your reasoning is clear, concise.
            '''

            >>> output = snail.get_cot_result(instruction, 2)

            >>> ds = snail.create_ds(instruction, output)

            >>> snail.transform_alpaca_format(ds)

            >>> snail.push_to_hf(json_path="path to file", repo_id="HF username/Repo name")
        
        Then you could use dataset as always:
            >>> ds = load_dataset("HF username/Repo name")

            >>> ds['train']

        Args:
            google_api_key (str): The API key for accessing Google's GenAI services.
            model_id (str): The identifier for the generative model to use.
            role (str): The expert role to define the context (e.g., 'political', 'technical') in system instruction for google search.
            user_query (str): The user's query to process.
            max_output_tokens (int): The maximum number of tokens in the generated output.

        Raises:
            ValueError: If any required parameter is missing, empty, or invalid.
        """
        # Assign input parameters to instance variables
        self.google_api_key = google_api_key
        self.model_id = model_id
        self.role = role
        self.user_query = user_query
        self.max_output_tokens = max_output_tokens

        # Validate input parameters
        if self.google_api_key is None or self.google_api_key == "":
            raise ValueError("API key is missing or empty")

        self.client = genai.Client(api_key=self.google_api_key)  # Initialize GenAI client
        self.google_search_tool = Tool(google_search=GoogleSearch())  # Set up Google Search tool

        if self.model_id is None or self.model_id == "":
            raise ValueError("Model ID is missing or empty")

        if self.role is None or self.role == "":
            raise ValueError("Role is missing or empty")

        if self.user_query is None or self.user_query == "":
            raise ValueError("User query is missing or empty")

        if self.max_output_tokens is None or self.max_output_tokens <= 0:
            raise ValueError(f"Max output tokens must be greater than zero, got {self.max_output_tokens}")

        # Define system instruction for Google Search tool using the provided role
        self.system_instruction_google_search = f"""You are {self.role} expert with 20 years of experience in this field.
        Over the course of your long career, you have done a lot of research on various topics related to your field.
        You have learnt how to do research as accurately as possible, while performing a maximum of 2 checks on your current results.
        Now you have this vast experience and are helping others to do the same.
        Begin answer only with final result.
        """

        # Define system instruction for Chain of Thought (CoT) processing
        self.system_instruction_cot = f"""You are a {self.role} expert skilled at explaining step by step mathematician problems, using a Chain of Thought (CoT) framework. Your response must include:
        - A thought process inside <thought></thought> tags, where you analyze the problem.
        - A final response inside <answer></answer> tags, solving the problem.
        Ensure your reasoning is clear, concise.
        """

    def searching(self) -> str:
        """
        Generate content based on the user query using the Google Search tool.

        Returns:
            str: The generated text response from the GenAI model.
        """
        try:
            response = self.client.models.generate_content(
            model=self.model_id,
            contents=self.user_query,
            config=GenerateContentConfig(
                tools=[self.google_search_tool],
                response_modalities=['TEXT'],
                system_instruction=self.system_instruction_google_search,
                max_output_tokens=self.max_output_tokens,
                )
            )
            return response.text
        except Exception as e:
            print(f"Error occurred while searching: {e}")

    @staticmethod
    def extract_listings(listings: str) -> List[str]:
        """
        Extract text after markers like \n1., \n2., etc., from the input text and return them as a list.
        Expected format is \n1. **Problem name** Problem text.

        Args:
            listings (str): The input text containing numbered markers (e.g., '\n1. text').

        Returns:
            List[str]: A list of strings, each being the text after a numbered marker.
        """
        # Use regex to find text following numbered markers and strip trailing whitespace
        quotes = [quote.rstrip() for quote in re.findall(r'\n\d+\.\s+(.*)', listings)]
        return quotes

    def get_cot_result(self, datas: List[str], delay: int = 5) -> List[str]:
        """
        Process a list of data entries using the Chain of Thought (CoT) framework and return results.

        Args:
            datas (List[str]): List of data entries (e.g., statements or quotes) to analyze.
            delay (int): Time in seconds to wait between processing each entry to avoid rate limits.

        Returns:
            List[str]: List of generated text responses for each data entry.
        """
        results = []
        for data in datas:
            try:
                # Generate content for each data entry using CoT configuration
                response = self.client.models.generate_content(
                    model=self.model_id,
                    config=GenerateContentConfig(
                        system_instruction=self.system_instruction_cot
                    ),
                    contents=data
                )
                if response:  # Ensure response is valid before appending
                    results.append(response.text)

                # Print detailed output for debugging or logging
                print(f"{'#'*140}\n\nData: {data}\n\nResponse: {response.text}\n\nUsage tokens: {response.usage_metadata}\n{'#'*140}")
                time.sleep(delay)  # Pause to respect rate limits
            except Exception as e:
                # Log errors with consistent f-string formatting
                print(f"Error occurred while processing data '{data}': {e}")
        return results

    def create_ds(self, instruction: List[str], output: List[str]) -> Dict[str, str]:
        """
        Create a dictionary by combining instruction and output lists.

        Args:
            instruction (List[str]): List of instruction strings.
            output (List[str]): List of corresponding output strings.

        Returns:
            Dict[str, str]: A dictionary mapping instructions to their outputs.

        Raises:
            ValueError: If the instruction and output lists have different lengths.
        """
        if len(instruction) != len(output):
            raise ValueError("Instruction and output must be the same length")

        return dict(zip(instruction, output))

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
        # Transform the data
        transformed_data = []
        for instruction, output in dataset.items():
            transformed_pair = {
                "instruction": instruction,
                "input": "",
                "output": output
            }
            transformed_data.append(transformed_pair)

        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'transformed_qa_{timestamp}.json'

        # Write the transformed data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, ensure_ascii=False, indent=2)

        return output_file, transformed_data

    def push_to_hf(self, json_path: str, repo_id: str) -> None:
        """
        Push a JSON dataset and its dataset card to a new Hugging Face dataset repository.

        Args:
            json_path (str): File path to the JSON file containing the dataset.
            repo_id (str): Repository identifier in the format 'username/reponame'.

        Raises:
            ValueError: If `json_path` or `repo_id` is empty or None.
        """

        if not json_path:
            raise ValueError("The JSON file path must be provided and cannot be empty.")
        if not repo_id:
            raise ValueError("The repository ID must be provided and cannot be empty.")

        try:
            url = create_repo(repo_id=repo_id, repo_type="dataset")
            username, _ = repo_id.split('/')
            
            content = DATASET_CARD.format(
                username = username,
            )
            card = DatasetCard(content)
            card.push_to_hub(repo_id)

            dataset = load_dataset("json", data_files=json_path)
            dataset.push_to_hub(repo_id)

            print(f"Congratulations on creating a new dataset {url}")
        except Exception as e:
            print(f"Error occurred while pushing dataset: {e}")