<div align="center">
    <img src="snail_mascot.png" width="200px" height="200px">
</div>


# Snail
### Generate CoT Dataset Using Gemini Model & Google Search Tool

This repository provides a powerful class designed to search for tasks, expressions, and similar entities that can be quantified, enabling you to generate a Chain-of-Thought (CoT) dataset with ease. By leveraging the Gemini model and the Google search tool, you can now seamlessly build datasets for advanced reasoning and problem-solving applications.

### Overview

The main objective of this class is to help you:
1. Search for specific tasks and expressions using Google search.
2. Extract enumerated listings from the search results.
3. Generate a comprehensive Chain-of-Thought dataset.
4. Create and push your dataset to Hugging Face in a streamlined manner.

### Getting Started

 ### 1. Initialize the Class
Set up the class by providing all the necessary parameters. One key parameter is role, which tailors the context for more relevant system instructions via system_instruction_google_search.
   ```python
   snail = Snail(google_api_key=google_api, model_id='gemini-2.0-flash-thinking-exp-01-21', user_query="List a 20 math problems from easiest to hard and numerate their", role="mathematician")
   result = snail.searching()
   ```
 ### 2. Extract Enumerations
After obtaining the results, extract the enumerated listings. Ensure that your user_query specifies the desired number of items to extract.
  ```python
  instruction = snail.extract_listings(result)
  ```
 ### 3. Customize Your CoT System Instruction
Tweak the system_instruction_cot parameter to better fit your needs. Your instruction should detail the problem-solving process step-by-step. The thought process should be encapsulated within <thought></thought> tags and the final answer within <answer></answer> tags.
  ```python
  snail.system_instruction_cot = f"""You are a {snail.role} expert skilled at explaining step by step mathematician problems, using a Chain of Thought (CoT) framework. Your response must include:
- A thought process inside <thought></thought> tags, where you analyze the problem.
- A final response inside <answer></answer> tags, solving the problem.
Ensure your reasoning is clear, concise.
 """
```
 ### 4. Generate CoT Results
Leverage the configured instructions to produce CoT outputs. This step generates responses in the desired CoT format.
  ```python   
  output = snail.get_cot_result(instruction, 2)
  ```
 ### 5. Create and Push Your Dataset
Finally, transform your outputs into a JSON-formatted dataset and push it to Hugging Face. 
  ```python
  ds = snail.create_ds(instruction, output)
  snail.transform_alpaca_format(ds)
  snail.push_to_hf(json_path="path to file", repo_id="HF username/Repo name")
  ```
