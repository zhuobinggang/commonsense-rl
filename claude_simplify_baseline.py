from claude_simplify import Claude_caller_simplify
from llm_simplify import Builder_old_style

class Claude_caller_baseline(Claude_caller_simplify):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.builder = Builder_old_style()

    def updated_description(self, description):
        return description
    
    def file_name_generate(self):
        shot = 'ZERO_SHOT'
        augment = 'AUGMENT_ON'
        filename_prefix = self.filename_prefix + '_' if self.filename_prefix else ''
        self.filename_raw = f'{filename_prefix}BASELINE_{shot}_{self.gpt_type}_{augment}_STEP_LIMIT_{self.step_limit}_{self.env.env.meta_info}'
        self.filename = self.filename_raw + '.pkl'
        print(self.filename_raw)

    def save_hook(self):
        pass