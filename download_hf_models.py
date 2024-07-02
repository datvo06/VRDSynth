import os
from huggingface_hub import hf_hub_download

# Repository ID
repo_id = "datvo06/fine-tuned-layoutlm"

# Get the current directory
output_dir = os.getcwd()

# Get the list of files in the repository
repo_files = hf_hub_download(repo_id=repo_id, filename=None, repo_type="model")

# Download each file from the repository
for file_path in repo_files:
    # Check if the file has a .bin extension
    if file_path.endswith(".bin"):
        # Download the file and overwrite if it already exists
        local_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="model",
            local_dir=output_dir,
            force_download=True,
        )
        
        print(f"Downloaded {file_path} to {local_file_path}")

print("All .bin files downloaded successfully!")
