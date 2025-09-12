import requests
from PIL import Image
from io import BytesIO
import base64
import json
import torch
from service.service_orchestrator import clip_model, clip_preprocess, get_shoe_collection
from config import settings
from openai import AzureOpenAI
import re
from utils.image_utils import get_base64_image_from_url

try:
    azure_client = AzureOpenAI(
        api_key=settings.openai_key,
        api_version="2024-02-15-preview",
        azure_endpoint=settings.openai_endpoint,
        default_headers={"Ocp-Apim-Subscription-Key": settings.openai_key}
    )
except Exception as e:
    print(f"Azure OpenAI init error: {e}")
    azure_client = None


async def extract_shoe_intent(user_query: str):
    if not azure_client:
        return {"query": user_query}
    try:
        system_content = settings.extract_shoe_intent_prompt
        resp = azure_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_query},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=250,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"extract_shoe_intent error: {e}")
        return {"query": user_query}


def get_clip_image_embedding(image_url: str):
    r = requests.get(image_url, timeout=10)
    r.raise_for_status()
    image = Image.open(BytesIO(r.content)).convert("RGB")
    image_input = clip_preprocess(image).unsqueeze(0)
    with torch.no_grad():
        emb = clip_model.encode_image(image_input).cpu().numpy().flatten().tolist()
    return emb


def get_clip_text_embedding(text: str):
    import clip
    tokens = clip.tokenize([text])
    with torch.no_grad():
        emb = clip_model.encode_text(tokens).cpu().numpy().flatten().tolist()
    return emb


def insert_to_vector_db(image_url: str, json_data: dict, description: str):
    col = get_shoe_collection()
    emb = get_clip_image_embedding(image_url)
    meta = {**json_data, "image_url": image_url}
    col.add(documents=[description], embeddings=[emb], ids=[image_url], metadatas=[meta])


def guess_brand(url: str):
    u = url.lower()
    if "beymen" in u: return "Beymen"
    if "boyner" in u: return "Boyner"
    if "lacoste" in u: return "Lacoste"
    return "Premium"

def image_to_json_and_caption(image_url: str):
    if azure_client is None:
        return {}, ""  # fallback
    b64 = get_base64_image_from_url(image_url)
    if not b64:
        return {}, ""  # fallback
    msgs = [
        {
            "role": "system",
            "content": "Ayakkabı görselini analiz et ve {json, caption} döndür. Sadece bu formatta JSON döndür.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this shoe image."},
                {"type": "image_url", "image_url": {"url": b64}},
            ],
        },
    ]
    try:
        r = azure_client.chat.completions.create(
            model="gpt-4o", messages=msgs, max_tokens=600, temperature=0
        )
        data = json.loads(r.choices[0].message.content)
        return data.get("json", {}), data.get("caption", "")
    except Exception as e:
        print(f"image_to_json_and_caption error: {e}")
        return {}, ""  # fallback


def process_images_to_json_and_insert(image_links: list[str]):
    print(f"Processing {len(image_links)} images...")
    ok, fail = 0, 0
    for url in image_links:
        try:
            jd, caption = image_to_json_and_caption(url)
            if not jd: jd = {}
            if not caption: caption = f"{guess_brand(url)} ayakkabı"
            insert_to_vector_db(url, jd, caption)
            ok += 1
            print(f"added: {url}")
        except Exception as e:
            fail += 1
            print(f"skip: {url} - {e}")
    print(f"Done. added={ok}, failed={fail}")


def _norm_txt(t: str) -> str:
    return re.sub(r"[^a-z0-9çğıöşü\s]", " ", (t or "").lower())

def _query_tokens(query: str) -> set[str]:
    qx = enrich_shoe_query(query)
    toks = set(_norm_txt(qx).split()) | set(_norm_txt(query).split())
    return {t for t in toks if len(t) >= 2}

def _meta_blob(meta: dict, doc: str) -> str:
    parts = [
        meta.get("name"), meta.get("brand"), meta.get("color"),
        meta.get("style"), meta.get("material"), meta.get("shoe_type"),
        meta.get("description"), doc or ""
    ]
    return _norm_txt(" ".join([p for p in parts if p]))

def _meta_score(tokens: set[str], blob: str) -> float:
    if not tokens or not blob:
        return 0.0
    hits = sum(1 for t in tokens if t in blob)
    return hits / max(1, len(tokens))

def _vec_score_from_distance(d: float) -> float:
    try:
        s = 1.0 - float(d)
        return max(0.0, min(1.0, s))
    except Exception:
        return 0.0

def vector_search_shoes(query: str, top_k: int = 5, weight_vector: float = 0.5):
    try:
        col = get_shoe_collection()
        q = enrich_shoe_query(query)
        emb = get_clip_text_embedding(q)
        res = col.query(
            query_embeddings=[emb],
            n_results=max(top_k * 8, top_k),
            include=["metadatas", "documents", "distances", "ids"],
        )

        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids   = res.get("ids", [[]])[0]
        if not docs:
            return "🔍 Veritabanımızda bu kriterlere uygun ayakkabı bulunamadı."

        tokens = _query_tokens(query)
        wv = max(0.0, min(1.0, weight_vector))
        wm = 1.0 - wv

        scored = []
        for d, _id, m, doc in zip(dists, ids, metas, docs):
            vs = _vec_score_from_distance(d)
            ms = _meta_score(tokens, _meta_blob(m or {}, doc))
            final = wv * vs + wm * ms
            scored.append((final, vs, ms, d, _id, m or {}, doc))

        ranked = sorted(scored, key=lambda x: (-x[0], str(x[4])))[:top_k]

        out = []
        for final, vs, ms, dist, _id, m, doc in ranked:
            name = m.get("name") or create_realistic_shoe_name(m)
            desc = (m.get("description") or doc) or ""
            attrs = build_shoe_attributes(m)
            img = m.get("image_url", "")
            out.append(
                f"## {name}\n\n{desc}\n\n{attrs}\n\nSkor: {final:.2f} (Vektör {vs:.2f} | Meta {ms:.2f}) | Mesafe: {dist:.3f}\n\n{f'![{name}]({img})' if img else ''}\n\n---"
            )

        return f"{len(out)} ayakkabı bulundu!\n\n" + "\n\n".join(out)
    except Exception as e:
        print(f"vector_search_shoes error: {e}")
        return "🔧 Arama sisteminde geçici bir sorun var."

def create_realistic_shoe_name(meta: dict):
    brand = meta.get("brand", "Premium")
    t = meta.get("shoe_type", "Ayakkabı")
    color = meta.get("color", "")
    parts = []
    if brand and brand != "Unknown":
        parts.append(brand)
    if color:
        parts.append(color.title())
    parts.append(t.title())
    return " ".join(parts)

def build_shoe_attributes(meta: dict):
    parts = []
    if meta.get("shoe_type"):
        parts.append(f"Tip: {meta['shoe_type'].title()}")
    if meta.get("color"):
        parts.append(f"Renk: {meta['color'].title()}")
    if meta.get("brand"):
        parts.append(f"Marka: {meta['brand']}")
    if meta.get("style"):
        parts.append(f"Stil: {meta['style'].title()}")
    if meta.get("material"):
        parts.append(f"Malzeme: {meta['material'].title()}")
    return " | ".join(parts) if parts else "Premium Kalite"

def enrich_shoe_query(query: str):
    q = query.lower()
    mapping = {
        "topuklu": "high heels stiletto",
        "şık": "elegant formal dress",
        "abiye": "evening formal dress",
        "spor": "athletic sneaker sport",
        "bot": "boots ankle boot",
        "yazlık": "summer light",
        "kışlık": "winter warm",
        "rahat": "comfortable casual",
        "formal": "formal dress business",
        "iş": "business formal office",
        "günlük": "casual daily wear",
        "yeşil": "green",
        "siyah": "black",
        "beyaz": "white",
        "kahverengi": "brown",
        "kırmızı": "red",
        "mavi": "blue",
        "taşlı": "embellished rhinestone crystal",
        "parlak": "shiny glittery metallic",
    }
    extra = [en for tr, en in mapping.items() if tr in q]
    return f"{query} {' '.join(extra)}" if extra else query


def get_current_weather(city: str):
    try:
        if getattr(settings, "accuwather_key", None):
            loc = requests.get(
                f"http://dataservice.accuweather.com/locations/v1/cities/search",
                params={"apikey": settings.accuwather_key, "q": city, "language": "tr-tr"},
                timeout=8,
            )
            loc.raise_for_status()
            items = loc.json()
            if items:
                key = items[0]["Key"]
                w = requests.get(
                    f"http://dataservice.accuweather.com/currentconditions/v1/{key}",
                    params={"apikey": settings.accuwather_key, "language": "tr-tr", "details": "true"},
                    timeout=8,
                )
                w.raise_for_status()
                cur = w.json()[0]
                return f"{city}: {cur['WeatherText']}, {cur['Temperature']['Metric']['Value']}°C (Hissedilen {cur['RealFeelTemperature']['Metric']['Value']}°C)"
        if getattr(settings, "weather_api_key", None):
            w = requests.get(
                "http://api.openweathermap.org/data/2.5/weather",
                params={"q": city, "appid": settings.weather_api_key, "units": "metric", "lang": "tr"},
                timeout=8,
            )
            if w.status_code == 200:
                d = w.json()
                return f"{city}: {d['weather'][0]['description']}, {d['main']['temp']}°C (Hissedilen {d['main']['feels_like']}°C), Nem {d['main']['humidity']}%"
        return f"{city} için hava durumu bilgisi alınamadı."
    except Exception as e:
        print(f"get_current_weather error: {e}")
        return "Hava durumu bilgisi alınırken hata oluştu."
    
def get_shoe_image_links():
    with open("./data/product_links.txt", "r") as file:
        return [line.strip() for line in file if line.strip()]
   