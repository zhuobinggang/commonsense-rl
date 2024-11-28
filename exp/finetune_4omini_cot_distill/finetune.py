from functools import lru_cache

E1 = 'ft:gpt-4o-mini-2024-07-18:personal::AY6Y2SEQ'
E2 = 'ft:gpt-4o-mini-2024-07-18:personal::AYR46lsF'
E3 = 'ft:gpt-4o-mini-2024-07-18:personal::AYSdQn8v'

@lru_cache(maxsize=None)
def get_client():
    from openai import OpenAI
    client = OpenAI()
    return client

# out: id='file-9C619S53TAisjJwmkTwbcE'
def create_file():
    client = get_client()
    return client.files.create(
        file=open("cot_distill_training.jsonl", "rb"),
        purpose="fine-tune"
    )

def fine_tune():
    client = get_client()
    return client.fine_tuning.jobs.create(
        training_file="file-AuBXZD2XLszydKeJSbVmkJ",
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={
            "n_epochs": 1
        }
    )

# 再次训练epoch2看看结果如何
def re_finetune():
    client = get_client()
    return client.fine_tuning.jobs.create(
        training_file="file-AuBXZD2XLszydKeJSbVmkJ",
        model=E2,
        hyperparameters={
            "n_epochs": 1
        }
    )