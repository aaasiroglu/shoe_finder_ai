import requests
from PIL import Image
from io import BytesIO
import base64

def get_base64_image_from_url(image_url: str):
    try:
        r = requests.get(image_url, timeout=10)
        r.raise_for_status()
        with Image.open(BytesIO(r.content)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"base64 error: {e}")
        return None