import chainlit as cl
from ctransformers import AutoModelForCausalLM


# llm = AutoModelForCausalLM.from_pretrained(
#     "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
# )

prompt = "capital city of India"

# print(llm(prompt))


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in short and concise way"
    prompt = f"### System:\n{system}\n\n### User:"
    if history is not None:
        prompt += f"This is conversation history: {''.join(history)}. Now answer the question."
    prompt += f"\n{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


@cl.on_message
async def main(message: cl.Message):
    prompt = get_prompt(message.content)
    response = llm(prompt)
    await cl.Message(
        response
    ).send()


@cl.on_chat_start
def on_chat_start():
    print("A new chat session has started!")
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )


'''
history = []

Question = "Which city is the capital of India?"

answer = ''

for word in llm(get_prompt(Question), stream=True):
    print(word, end='', flush=True)
    answer += word
print()

history.append(answer)

Question = "And which is of the United States?"

for word in llm(get_prompt(Question, history), stream=True):
    print(word, end='', flush=True)
print()
'''
