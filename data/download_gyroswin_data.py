# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "paramiko",
#     "python-dotenv",
#     "tqdm",
# ]
# ///

# run with: uv run --env-file .env.local .\data\download_gyroswin_data.py

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


REMOTE_BASE_PATH = Path("/restricteddata/ukaea/gyrokinetics/raw")
LOCAL_PATH = Path(__file__).parent.resolve() / "flux" / "raw"
REMOTE_FLUX_FILE_NAME: str = "fluxes.dat"
REMOTE_INPUT_FILE_NAME: str = "input.dat"
LOCAL_FLUX_FILE_CONVENTION: str = "fluxes_{iteration}.dat"
LOCAL_INPUT_FILE_CONVENTION: str = "input_{iteration}.dat"

GYROSWIN_ID_BENCHMARK_IDXS: list[int] = [8, 115, 131, 148, 235, 262]
GYROSWIN_VALIDATION_IDXS: list[int] = [13, 100, 200]
# training_trajectories: iteration_{0-7,9-12,14-99,101-114,116-130,132-147,149-199,201-234,236-261,263-299}.h5
GYROSWIN_TRAIN_IDXS = set(
    [
        *list(range(0, 8)),
        *list(range(9, 13)),
        *list(range(14, 100)),
        *list(range(101, 115)),
        *list(range(116, 131)),
        *list(range(132, 148)),
        *list(range(149, 200)),
        *list(range(201, 235)),
        *list(range(236, 262)),
        *list(range(263, 300)),
    ]
)

GYROSWIN_TRAIN_PATH = LOCAL_PATH / "gyroswin_train"
GYROSWIN_ID_PATH = LOCAL_PATH / "gyroswin_id"
GYROSWIN_OOD_PATH = LOCAL_PATH / "gyroswin_ood"
GYROSWIN_VAL_PATH = LOCAL_PATH / "gyroswin_val"

# Ensure local folder exists
os.makedirs(LOCAL_PATH, exist_ok=True)
os.makedirs(GYROSWIN_TRAIN_PATH, exist_ok=True)
os.makedirs(GYROSWIN_ID_PATH, exist_ok=True)
os.makedirs(GYROSWIN_OOD_PATH, exist_ok=True)
os.makedirs(GYROSWIN_VAL_PATH, exist_ok=True)

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


# List iteration folders
folders = [
    f
    for f in sftp.listdir(REMOTE_BASE_PATH.as_posix())
    if f.startswith("iteration_")
    and not f.endswith("Lin")
    and not f.endswith("nodumps")
]

gyroswin_train_idx_counter = 1000
gyroswin_val_idx_counter = 2000
gyroswin_id_idx_counter = 3000
for folder in tqdm(folders):
    idx = folder.split("_")[-1]
    try:
        idx = int(idx)
    except ValueError:
        tqdm.write(
            f"Could not parse iteration index from folder name {folder}, skipping."
        )
        continue

    if idx in GYROSWIN_ID_BENCHMARK_IDXS:
        local_subfolder = GYROSWIN_ID_PATH
        iteration = gyroswin_id_idx_counter
        gyroswin_id_idx_counter += 1
    elif idx in GYROSWIN_VALIDATION_IDXS:
        local_subfolder = GYROSWIN_VAL_PATH
        iteration = gyroswin_val_idx_counter
        gyroswin_val_idx_counter += 1
    elif idx in GYROSWIN_TRAIN_IDXS:
        local_subfolder = GYROSWIN_TRAIN_PATH
        iteration = gyroswin_train_idx_counter
        gyroswin_train_idx_counter += 1
    else:
        tqdm.write(f"Iteration index {idx} not in any known split, skipping.")
        continue

    remote_folder: Path = REMOTE_BASE_PATH / folder

    # Download fluxes.dat
    remote_flux_file = remote_folder / REMOTE_FLUX_FILE_NAME
    local_flux_file = local_subfolder / LOCAL_FLUX_FILE_CONVENTION.format(
        iteration=iteration
    )
    try:
        tqdm.write(f"Downloading {remote_flux_file} -> {local_flux_file}")
        sftp.get(remote_flux_file.as_posix(), local_flux_file.as_posix())
    except IOError:
        tqdm.write(f"File {remote_flux_file} does not exist, skipping.")

    # Download input.dat
    remote_input_file = remote_folder / REMOTE_INPUT_FILE_NAME
    local_input_file = local_subfolder / LOCAL_INPUT_FILE_CONVENTION.format(
        iteration=iteration
    )
    try:
        tqdm.write(f"Downloading {remote_input_file} -> {local_input_file}")
        sftp.get(remote_input_file.as_posix(), local_input_file.as_posix())
    except IOError:
        tqdm.write(f"File {remote_input_file} does not exist, skipping.")


REMOTE_BASE_PATH_OOD = REMOTE_BASE_PATH / "ood"
folders = [
    f
    for f in sftp.listdir(REMOTE_BASE_PATH_OOD.as_posix())
    if f.startswith("iteration_")
    and not f.endswith("Lin")
    and not f.endswith("nodumps")
]
gyroswin_ood_idx_counter = 4000
for folder in tqdm(folders, desc="OOD folders"):
    remote_folder: Path = REMOTE_BASE_PATH_OOD / folder

    # Download fluxes.dat
    remote_flux_file = remote_folder / REMOTE_FLUX_FILE_NAME
    local_flux_file = GYROSWIN_OOD_PATH / LOCAL_FLUX_FILE_CONVENTION.format(
        iteration=gyroswin_ood_idx_counter
    )
    try:
        tqdm.write(f"Downloading {remote_flux_file} -> {local_flux_file}")
        sftp.get(remote_flux_file.as_posix(), local_flux_file.as_posix())
    except IOError:
        tqdm.write(f"File {remote_flux_file} does not exist, skipping.")

    # Download input.dat
    remote_input_file = remote_folder / REMOTE_INPUT_FILE_NAME
    local_input_file = GYROSWIN_OOD_PATH / LOCAL_INPUT_FILE_CONVENTION.format(
        iteration=gyroswin_ood_idx_counter
    )
    try:
        tqdm.write(f"Downloading {remote_input_file} -> {local_input_file}")
        sftp.get(remote_input_file.as_posix(), local_input_file.as_posix())
    except IOError:
        tqdm.write(f"File {remote_input_file} does not exist, skipping.")

    gyroswin_ood_idx_counter += 1

# Close connections
sftp.close()
ssh.close()
print("Download complete!")
