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
def create_file(file_path = "out.jsonl"):
    client = get_client()
    return client.files.create(
        file=open(file_path, "rb"),
        purpose="fine-tune"
    )

def fine_tune(file_id_openai = 'file-9C619S53TAisjJwmkTwbcE', n_epochs = 3):
    client = get_client()
    return client.fine_tuning.jobs.create(
        training_file=file_id_openai,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={
            "n_epochs": n_epochs
        }
    )

def retune(file_id_openai, model_id_openai):
    client = get_client()
    return client.fine_tuning.jobs.create(
        training_file=file_id_openai,
        model=model_id_openai,
        hyperparameters={
            "n_epochs": 1
        }
    )