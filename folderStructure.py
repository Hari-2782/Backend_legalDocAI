import os

# Define the base directory
base_dir = "Backend_legalDocAI"

# Define the folder structure as a list of paths
folders_to_create = [
    "app",
    "app/routes",
    "app/services",
    "app/utils",
    "retrain",
    "uploads",
    "chroma_db"
]

# Define the file structure as a list of paths
files_to_create = [
    "app/__init__.py",
    "app/main.py",
    "app/config.py",
    "app/models.py",
    "app/database.py",
    "app/routes/__init__.py",
    "app/routes/upload.py",
    "app/routes/qa.py",
    "app/routes/feedback.py",
    "app/routes/summarize.py",
    "app/routes/retrain.py",
    "app/services/__init__.py",
    "app/services/pdf_parser.py",
    "app/services/embedding.py",
    "app/services/rag.py",
    "app/services/inference.py",
    "app/services/feedback.py",
    "app/services/retrain.py",
    "app/utils/__init__.py",
    "app/utils/logger.py",
    "app/utils/fileops.py",
    "app/utils/hf_client.py",
    "retrain/dataset_builder.py",
    "retrain/train_lora.py",
    "retrain/config.json",
    "firebase_key.json",
    "requirements.txt",
    "Dockerfile",
    "README.md"
]

# Create the base directory
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Change to the base directory
os.chdir(base_dir)

# Create folders
for folder in folders_to_create:
    os.makedirs(folder, exist_ok=True)

# Create files
for file in files_to_create:
    with open(file, 'w') as f:
        pass  # Create an empty file

print("Folder structure and files created successfully.")