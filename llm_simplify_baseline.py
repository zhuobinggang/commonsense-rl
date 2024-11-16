from llm_simplify import GPT_Caller_Simplify, Builder_old_style

class GPT_Caller_Baseline(GPT_Caller_Simplify):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.builder = Builder_old_style() # full action list

    def updated_description(self, description):
        return description # full description
    
    def file_name_generate(self):
        shot = 'ZERO_SHOT'
        augment = 'AUGMENT_ON'
        filename_prefix = self.filename_prefix + '_' if self.filename_prefix else ''
        self.filename_raw = f'{filename_prefix}BASELINE_{shot}_{self.gpt_type}_{augment}_STEP_LIMIT_{self.step_limit}_{self.env.env.meta_info}'
        self.filename = self.filename_raw + '.pkl'
        print(self.filename_raw)

    def save_hook(self):
        pass

class GPT_caller_simple_desc_only(GPT_Caller_Simplify):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.builder = Builder_old_style() # full action list

    def file_name_generate(self):
        shot = 'ZERO_SHOT'
        augment = 'AUGMENT_ON'
        filename_prefix = self.filename_prefix + '_' if self.filename_prefix else ''
        self.filename_raw = f'{filename_prefix}SIMPLEDESC_{shot}_{self.gpt_type}_{augment}_STEP_LIMIT_{self.step_limit}_{self.env.env.meta_info}'
        self.filename = self.filename_raw + '.pkl'
        print(self.filename_raw)

    def save_hook(self):
        pass
