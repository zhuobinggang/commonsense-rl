import global_variable as G
from functools import lru_cache
from claude_caller import Prompt_builder, Builder1, command_from_text, Claude_Caller


@lru_cache(maxsize=None)
def get_client():
    from openai import OpenAI
    client = OpenAI()
    return client


def text_from_raw_response(completion):
    content = completion.choices[0].message.content
    return content


def quest_gpt_raw(client, system_msg, user_msg, gpt_type, need_print = False):
    completion = client.chat.completions.create(
        model=gpt_type,  # 
        messages=[{
            "role": "system",
            "content": system_msg
        }, {
            "role": "user",
            "content": user_msg
        }],
        temperature=0)
    # To clipboard
    usage = str(completion.usage)
    # To clipboard
    # NOTE: 自动提取command
    text = text_from_raw_response(completion)
    if need_print:
        print(text)
    dic = {'response': text, 'usage': usage}
    the_command = command_from_text(text) # Might be None
    return completion, dic, the_command


class GPT_Caller(Claude_Caller):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def quest_my_llm(self, system_msg, user_msg, llm_type, need_print = False):
        complete, dic, the_command = quest_gpt_raw(get_client(), system_msg,
                                        user_msg,
                                        llm_type, need_print=need_print) # 获得the_command，可能为空
        return complete, dic, the_command

