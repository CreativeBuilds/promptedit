import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def filter_result(result) -> str:
    result = result.replace("\n", " ")
    result = result[result.rfind("AGENT:") + 7:]
    result = result.split("  ")[0]
    result = result.split("```")[0]
    result = result.strip()
    return result

def read_sample_prompts(path: str) -> str:
    output = ""

    # Sort files by their name to ensure they are processed in order
    files = sorted(os.listdir(path))

    # Extract and pair '.in' and '.out' files
    input_files = [f for f in files if f.endswith('.in')]
    output_files = [f for f in files if f.endswith('.out')]

    # Ensure each input file has a corresponding output file
    if len(input_files) != len(output_files):
        print("Mismatch between the number of input and output files!")
    else:
        for idx, (in_file, out_file) in enumerate(zip(input_files, output_files)):
            # Read input file
            with open(os.path.join(path, in_file), 'r') as f:
                prompt = f.read().strip()

            # Read output file
            with open(os.path.join(path, out_file), 'r') as f:
                agent_response = f.read().strip()

            # Append to the output string in the desired format
            output += f"PROMPT: {prompt}\n"
            output += f"AGENT: masterpiece:{idx+1}, {agent_response}\n\n"

    return output


def main(prompt_in: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained("cerebras/btlm-3b-8k-base")
    model = AutoModelForCausalLM.from_pretrained("cerebras/btlm-3b-8k-base", trust_remote_code=True, torch_dtype="auto").to("cuda")

    sample_prompts = read_sample_prompts('SDXL_prompts')
    prompt = f"""SYSTEM: You are an agent tasked with creating better, more descriptive image prompts that result in higher quality outputs. You are given a user prompt, and determine how, if it all, to change the users image. Do you understand?
AGENT: Yes, I understand.
SYSTEM: Great. Below there will be a set of examples. For each example, you will be given the users prompt and the expected output. You will be then given a new user prompt and have to generate a new output. Do not output anything beyond the prompt. Do you understand?
AGENT: Yes.
SYSTEM: Another rule, do not list numbers in your response.
AGENT: Yes, I understand.

{sample_prompts}

PROMPT: {prompt_in}
AGENT:"""

    # Change the last prompt to whatever you want to test
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


    # Generate text using the model
    outputs = model.generate(
        **inputs,
        num_beams=5,
        max_new_tokens=50,
        early_stopping=True,
        no_repeat_ngram_size=2,
        # use random seed
        top_k=10,
        top_p=0.9,
        temperature=0.9,
        do_sample=True,
    )
    # Convert the generated token IDs back to text
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_text[0]

if __name__ == "__main__":
    edited = main("A picture of a dog")
    print(filter_result(edited))