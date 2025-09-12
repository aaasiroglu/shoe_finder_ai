import clip
import torch, random, numpy as np
from chromadb import PersistentClient

clip_model, clip_preprocess = clip.load("ViT-B/32")
chroma_client = PersistentClient(path="./chroma_db")

def get_shoe_collection():
    return chroma_client.get_or_create_collection(name="shoe_images")

def get_hotel_collection():
    return get_shoe_collection()

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)