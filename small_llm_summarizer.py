from llm_caller import get_client, GPT_Caller

GPT4O = 'gpt-4o-2024-08-06'
GPT4OMINI = 'gpt-4o-mini-2024-07-18'

class Prompt_builder:

    def __init__(self):
        self.system_msg = ''
        self.user_msg = ''
        self.prompt = ''
        self.desc = ''

    def build(self):
        system_msg = ''
        system_msg += 'Reform the environment description using format: Room[Furniture[Object]]\n'
        system_msg += 'Example: Bedroom[wardrobe, chest of drawers[black sock], desk[pen, eraser]]'
        system_msg = system_msg.strip() + '\n'
        self.system_msg = system_msg
        user_msg = ''
        user_msg += 'The environment description:\n'
        user_msg += self.desc.strip().replace('\n', '')
        user_msg = user_msg.strip() + '\n'
        self.user_msg = user_msg
        self.prompt = f'{system_msg}{user_msg}'

    def sys_usr_msg(self):
        return self.system_msg, self.user_msg


def quest_summarization(system_msg,
                        user_msg,
                        gpt_type=GPT4OMINI):
    client = get_client()
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
    content = completion.choices[0].message.content
    usage = str(completion.usage)
    return content


class GPT_Caller_Simple_Desc(GPT_Caller):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summarize_prompt_builder = Prompt_builder()
        self.summarize_log = ''

    def updated_description(self, description):
        self.summarize_prompt_builder.desc = description
        self.summarize_prompt_builder.build()
        sys_msg, usr_msg = self.summarize_prompt_builder.sys_usr_msg()
        new_desc = quest_summarization(sys_msg, usr_msg)
        self.summarize_log += f'{self.summarize_prompt_builder.prompt}\n\n{new_desc}\n\n'
        return new_desc
        
    def save_hook(self):
        env = self.env.env
        filename = 'exp/auto_filename/' + self.filename_raw + f'score_{env.last_reward}_summarization_log.txt'
        f = open(filename, 'w')
        f.write(self.summarize_log)
        f.close()

