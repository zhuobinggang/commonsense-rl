from llm_simplify import GPT_Caller_Simplify, json_obj_from_text
from claude_caller import get_client, text_from_raw_response, usage_from_raw_response

CLAUDE = 'claude-3-5-sonnet-20241022'

def quest_claude_simple(client, system, user, claude_type = "claude-3-5-sonnet-20241022", max_tokens = 500, verbose = False):
    completion = client.messages.create(
        model=claude_type,
        max_tokens=max_tokens,
        temperature=0,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user
                    }
                ]
            }
        ]
    )
    text = text_from_raw_response(completion)
    obj = json_obj_from_text(text)
    if verbose:
        print(text)
        print('------------------>')
        print(obj)
    dic = {'response': obj, 'usage': usage_from_raw_response(completion)}
    the_command = obj['action'] # Might be None
    return completion, dic, the_command


class Claude_caller_simplify(GPT_Caller_Simplify):
    def quest_my_llm(self, system_msg, user_msg, llm_type, verbose = False):
        complete, dic, the_command = quest_claude_simple(get_client(), system_msg,
                                        user_msg,
                                        llm_type, verbose=verbose) # 获得the_command，可能为空
        return complete, dic, the_command