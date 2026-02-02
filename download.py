from openpi.shared import download

# 会自动下载到 assets/pi05_base/
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
print("Checkpoint downloaded to:", checkpoint_dir)
