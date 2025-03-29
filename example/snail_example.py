from getpass import getpass
from snail.snail import Snail

google_api_key: str = getpass("Enter Google api key")
model_id: str = "gemini-2.0-flash-thinking-exp-01-21"
user_query: str = "List 3 hard problems in physics, list them"
role: str = "theoretical physicist"
delay: int = 5
repo_id: str = ""

# Do not forget run this `huggingface-cli login`

def main():
    snail = Snail(google_api_key=google_api_key, model_id=model_id, user_query=user_query, role=role)

    result_of_serching = snail.searching()

    print(f"Result of searching: \n\n{result_of_serching}")

    instruction = snail.extract_listings(result_of_serching)

    print(f"The instructions: \n\n{instruction}")

    # Or completely rewrite 
    snail.system_instruction_cot = f'''You are a {role} expert skilled at explaining difficult problems related to physics step by step, using a Chain of Thought (CoT) framework. Your response must include:
    - A thought process inside <thought></thought> tags, where you analyze the problem.
    - A final response inside <answer></answer> tags, solving the problem.
    Ensure your reasoning is clear and concise.
    '''

    output = snail.get_cot_result(instruction, delay=delay)

    ds = snail.create_ds(instruction, output)

    print(f"The dataset: \n\n{ds}")

    saved_json_file, transformed_ds = snail.transform_alpaca_format(ds)

    print(f"Dataset preview: \n\n{transformed_ds}")

    snail.push_to_hf(json_path=saved_json_file, repo_id=repo_id)


if __name__ == "__main__":
    main()