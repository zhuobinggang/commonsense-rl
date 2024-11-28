from functools import lru_cache

E1 = 'ft:gpt-4o-mini-2024-07-18:personal::AXLvStgP'
E2 = 'ft:gpt-4o-mini-2024-07-18:personal::AYRZHi84'
E3 = 'ft:gpt-4o-mini-2024-07-18:personal::AYS1pBmz'

@lru_cache(maxsize=None)
def get_client():
    from openai import OpenAI
    client = OpenAI()
    return client

# out: id='file-9C619S53TAisjJwmkTwbcE'
def create_file():
    client = get_client()
    return client.files.create(
        file=open("out.jsonl", "rb"),
        purpose="fine-tune"
    )

def fine_tune():
    client = get_client()
    return client.fine_tuning.jobs.create(
        training_file="file-9C619S53TAisjJwmkTwbcE",
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={
            "n_epochs": 1
        }
    )

def retune():
    client = get_client()
    return client.fine_tuning.jobs.create(
        training_file="file-9C619S53TAisjJwmkTwbcE",
        model=E2,
        hyperparameters={
            "n_epochs": 1
        }
    )