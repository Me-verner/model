import os
from huggingface_hub import snapshot_download, HfApi
from tqdm import tqdm
import getpass

# مسیر اصلی کش داخل Volume
BASE_CACHE_DIR = "/runpod-volume"
os.makedirs(BASE_CACHE_DIR, exist_ok=True)

# دریافت امن توکن از کاربر
HF_TOKEN = getpass.getpass("🔐 Enter your Hugging Face token: ")

# تست اعتبار توکن
api = HfApi()
try:
    user = api.whoami(token=HF_TOKEN)
    print(f"🔐 Logged in as: {user['name']}")
except Exception as e:
    print(f"❌ Invalid token: {e}")
    exit(1)

MODELS = [
    "HiDream-ai/HiDream-I1-Full",
    "HiDream-ai/HiDream-I1-Dev",
    "HiDream-ai/HiDream-I1-Fast",
    "azaneko/HiDream-I1-Full-nf4",
    "azaneko/HiDream-I1-Dev-nf4",
    "azaneko/HiDream-I1-Fast-nf4",
    "meta-llama/Llama-3.1-8B-Instruct"
]

print(f"\n📦 Downloading models into separate folders in: {BASE_CACHE_DIR}\n")

for model_id in MODELS:
    model_folder = model_id.replace("/", "--")  # ساخت نام پوشه امن
    output_path = os.path.join(BASE_CACHE_DIR, model_folder)

    print(f"🔽 Downloading {model_id} → {output_path}")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=output_path,
            token=HF_TOKEN,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack"],
            tqdm_class=tqdm
        )
        print(f"✅ Finished: {model_id}\n")
    except Exception as e:
        print(f"❌ Failed to download {model_id}: {e}\n")

print("🎉 All models downloaded and organized.")
