import os
from huggingface_hub import HfApi, list_repo_files

# Set up the Hugging Face Hub API
api = HfApi()

# Repository ID
repo_id = "datvo06/fine-tuned-layoutlm"

# Find all .bin files in the current directory and its subdirectories
bin_files = []
repo_files = list_repo_files(repo_id=repo_id, repo_type='model')
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".bin") or file.startswith('stage3'):
            # Check if the relative path is already in repo_files
            relative_path = os.path.relpath(os.path.join(root, file), ".")
            if relative_path not in repo_files:
                bin_files.append(os.path.join(root, file))

# Upload each .bin file to the Hugging Face Hub
for file_path in bin_files:
    # Get the relative path of the file
    relative_path = os.path.relpath(file_path, ".")
    
    # Upload the file to the repository
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=relative_path,
        repo_id=repo_id,
        repo_type="model",
    )
    
    print(f"Uploaded {file_path} to {repo_id}/{relative_path}")

print("All .bin files uploaded successfully!")
