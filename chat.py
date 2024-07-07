import chainlit as cl
from ctransformers import AutoModelForCausalLM


# llm = AutoModelForCausalLM.from_pretrained(
#     "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
# )

prompt = "capital city of India"

# print(llm(prompt))


def get_prompt_orca(instruction: str, history: list[str] | None = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in short and concise way"
    prompt = f"### System:\n{system}\n\n### User:"
    if len(history) > 0:
        prompt += f"This is conversation history: {''.join(history)}. Now answer the question."
    if instruction == 'forget everything':
        history = []
        prompt = ''
        return prompt
    prompt += f"\n{instruction}\n\n### Response:\n"
    return prompt


def get_prompt_llama2(instruction: str, history: list[str] | None = None) -> str:
    system = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. Also ensure that your answers are pinpoint, accurate and concise as asked. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
    if len(history) > 0:
        prompt += f"This is conversation history: {''.join(history)}. Now answer the question."
    if instruction == 'forget everything':
        history = []
        prompt = ''
        return prompt
    prompt += f"\n{instruction} [/INST]"
    return prompt


def select_llm(llm_name: str):
    global llm, get_prompt
    if llm_name == "use llama2":
        llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
        )
        get_prompt = get_prompt_llama2
        return "Model Llama is running"
    elif llm_name == "use orca":
        llm = AutoModelForCausalLM.from_pretrained(
            "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
        )
        get_prompt = get_prompt_orca
        return "Model Orca is running"
    else:
        return "Model not found"
    

@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content='')
    await msg.send()
    
    global llm, get_prompt

    if message.content == 'use orca':
        response = select_llm(message.content)
        await cl.Message(response).send()
        return
    elif message.content == 'use llama2':
        response = select_llm(message.content)
        await cl.Message(response).send()
        return

    prompt = get_prompt(message.content, message_history)
    
    answer = ''
    if prompt == '':
        message_history = []
        str = "Uh oh, I've just forgotten our conversation history"
        for word in str.split():
            if word == str.split()[-1]:
                str = word
            else:
                str = word + ' '
            await msg.stream_token(str)
        await msg.update()

    else:
        for word in llm(prompt, stream=True):
            await msg.stream_token(word)
            answer += word
        await msg.update()
    message_history.append(answer)


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set('message_history', [])
    print("A new chat session has started!")
    global llm

    select_llm('use orca')
    


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
