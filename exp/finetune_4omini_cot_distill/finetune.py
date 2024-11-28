from functools import lru_cache

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