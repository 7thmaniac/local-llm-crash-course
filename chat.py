from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

prompt = "capital city of India"

# print(llm(prompt))


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in short and concise way"
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


Question = "Which city is the capital of India?"


for word in llm(get_prompt(Question), stream=True):
    print(word, end='', flush=True)
print()
