from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q4_K_M.gguf", model_type="llama", gpu_layers=0)

# print(llm("AI is going to"))

Question = "What is the name of capital city of India? Please give only name of the city nothing else is needed."


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. Also ensure that your answers are pinpoint, accurate and concise as asked. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    return prompt


for word in llm(get_prompt(Question), stream=True):
    print(word, end='', flush=True)
print()
