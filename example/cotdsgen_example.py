from getpass import getpass
from snail.cot_dsgen import CoTDatasetGenerator

from rich.console import Console
from rich.panel import Panel

# Set up Rich console for pretty prints
console = Console()

google_api_key: str = getpass("Enter Google api key: ")
model_id: str = "gemini-2.0-flash-thinking-exp-01-21"
user_query: str = "List 3 hard problems in physics, list them"
role: str = "theoretical physicist"
delay: int = 5
repo_id: str = ""

# Do not forget to run `!huggingface-cli login` to push ds to HF

def main():
    snail = CoTDatasetGenerator(google_api_key=google_api_key, model_id=model_id, user_query=user_query, role=role)

    result_of_serching = snail.searching()

    console.print(Panel("Result of searching:", style="green"))
    console.print(Panel(result_of_serching, title="CoT Processing Output", style="red"))

    instruction = snail.extract_listings(result_of_serching)

    console.print(Panel("The instruction: ", style="green"))
    for i in instruction:
        console.print(Panel(i, title="Instruction output", style="green"))

    # Or completely rewrite
    snail.system_instruction_cot = f'''You are a {role} expert skilled at explaining difficult problems related to physics step by step, using a Chain of Thought (CoT) framework. Your response must include:
    - A thought process inside <thought></thought> tags, where you analyze the problem.
    - A final response inside <answer></answer> tags, solving the problem.
    Ensure your reasoning is clear, concise.
    '''

    console.print(Panel("Cot output: ", style="green"))
    output = snail.get_result(instruction, 2)

    ds = snail.create_ds(instruction, output)

    saved_json_file, transformed_ds = snail.transform_alpaca_format(ds)

    console.print(Panel("Dataset preview: ", style="green"))
    for i in transformed_ds:
        panel_content = (
            f"[bold green]Instruction:[/bold green] {i['instruction']}\n\n"
            f"[bold bold]Input:[/bold bold] {i['input']}\n\n"
            f"[bold red]Output:[/bold red] {i['output']}"
        )
        console.print(Panel(panel_content, title="Dataset preview", expand=False))

    snail.push_to_hf(json_path=saved_json_file, repo_id=repo_id)


if __name__ == "__main__":
    main()