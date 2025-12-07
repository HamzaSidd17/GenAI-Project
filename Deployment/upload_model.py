from huggingface_hub import HfApi, create_repo

model_path = "./final_model_distilbert-base-uncased"
repo_name = "Hamza-003/ai-text-detector-distilbert"

api = HfApi()
create_repo(repo_name, exist_ok=True, repo_type="model")
api.upload_folder(
    folder_path=model_path,
    repo_id=repo_name,
    repo_type="model"
)
print(f"✅ Model uploaded to: https://huggingface.co/{repo_name}")
