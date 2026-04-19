import kagglehub
import shutil
import os

def download_dataset():
    print(" Downloading healthcare dataset from Kaggle...")

    # Download using kagglehub
    path = kagglehub.dataset_download("prasad22/healthcare-dataset")
    print(f" Dataset downloaded to: {path}")

    # Find the CSV file
    for file in os.listdir(path):
        if file.endswith(".csv"):
            source = os.path.join(path, file)
            destination = os.path.join("data", "raw", "healthcare.csv")

            # Make sure the folder exists
            os.makedirs("data/raw", exist_ok=True)

            # Copy it into our project
            shutil.copy(source, destination)
            print(f" Saved to: {destination}")
            return destination

    print(" No CSV file found in download.")
    return None

if __name__ == "__main__":
    download_dataset()