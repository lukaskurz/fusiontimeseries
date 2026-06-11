# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "paramiko",
#     "python-dotenv",
#     "tqdm",
# ]
# ///

# run with: uv run --env-file .env.local .\data\download_batches.py

import os

from tqdm import tqdm  # type: ignore
from dotenv import load_dotenv  # type: ignore
import paramiko  # type: ignore
from pathlib import Path

# Load sensitive info
load_dotenv()
REMOTE_USER = os.getenv("REMOTE_USER")
if not REMOTE_USER:
    raise ValueError("REMOTE_USER not set in environment variables")

REMOTE_HOST = os.getenv("REMOTE_HOST")
if not REMOTE_HOST:
    raise ValueError("REMOTE_HOST not set in environment variables")

REMOTE_PORT_STR = os.getenv("REMOTE_PORT")
if not REMOTE_PORT_STR:
    raise ValueError("REMOTE_PORT not set in environment variables")
REMOTE_PORT = int(REMOTE_PORT_STR)

PASSWORD = os.getenv("PASSWORD")  # optional if using key-based auth
if not PASSWORD:
    raise ValueError("PASSWORD not set in environment variables")

REMOTE_BASE_PATH = Path("/restricteddata/ukaea/gyrokinetics/raw/new_data")
LOCAL_BASE_PATH = Path(__file__).parent.resolve() / "flux" / "raw"
REMOTE_FLUX_FILE_NAME: str = "fluxes.dat"
REMOTE_INPUT_FILE_NAME: str = "input.dat"
LOCAL_FLUX_FILE_CONVENTION: str = "fluxes_{iteration}.dat"
LOCAL_INPUT_FILE_CONVENTION: str = "input_{iteration}.dat"

# Create SSH client
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(
    REMOTE_HOST,
    port=REMOTE_PORT,
    username=REMOTE_USER,
    password=PASSWORD,
)

# Open SFTP session
sftp = ssh.open_sftp()

# List batch folders
batch_folders = [
    f for f in sftp.listdir(REMOTE_BASE_PATH.as_posix()) if f.startswith("batch_")
]

for batch_folder in tqdm(batch_folders, desc="Batches"):
    remote_batch_path = REMOTE_BASE_PATH / batch_folder
    local_batch_path = LOCAL_BASE_PATH / batch_folder

    # Ensure local batch folder exists
    os.makedirs(local_batch_path, exist_ok=True)

    # List iteration folders within this batch
    try:
        iteration_folders = [
            f
            for f in sftp.listdir(remote_batch_path.as_posix())
            if f.startswith("iteration_") and not f.endswith("Lin")
        ]
    except IOError:
        tqdm.write(f"Could not access {remote_batch_path}, skipping.")
        continue

    for iteration_folder in tqdm(
        iteration_folders, desc=f"{batch_folder}", leave=False
    ):
        iteration = iteration_folder.split("_")[-1]
        remote_iteration_path = remote_batch_path / iteration_folder

        # Download fluxes.dat
        remote_flux_file = remote_iteration_path / REMOTE_FLUX_FILE_NAME
        local_flux_file = local_batch_path / LOCAL_FLUX_FILE_CONVENTION.format(
            iteration=iteration
        )
        try:
            tqdm.write(f"Downloading {remote_flux_file} -> {local_flux_file}")
            sftp.get(remote_flux_file.as_posix(), local_flux_file.as_posix())
        except IOError:
            tqdm.write(f"File {remote_flux_file} does not exist, skipping.")

        # Download input.dat
        remote_input_file = remote_iteration_path / REMOTE_INPUT_FILE_NAME
        local_input_file = local_batch_path / LOCAL_INPUT_FILE_CONVENTION.format(
            iteration=iteration
        )
        try:
            tqdm.write(f"Downloading {remote_input_file} -> {local_input_file}")
            sftp.get(remote_input_file.as_posix(), local_input_file.as_posix())
        except IOError:
            tqdm.write(f"File {remote_input_file} does not exist, skipping.")


# Close connections
sftp.close()
ssh.close()
print("Download complete!")
