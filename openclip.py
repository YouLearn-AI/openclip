from io import BytesIO
import base64
import torch

try:
    import open_clip
    from PIL import Image
    import torchvision
except ImportError as e:
    print(f"Required libraries are missing: {e}. Install them before using this class.")

class OpenCLIPEmbeddings:
    _instances = {}
    model_name: str
    checkpoint: str
    model : open_clip.model.CLIP
    preprocess : torchvision.transforms.transforms.Compose
    tokenizer : open_clip.tokenizer.SimpleTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __new__(cls, model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k"):
        key = (model_name, checkpoint)
        if key not in cls._instances:
            instance = super(OpenCLIPEmbeddings, cls).__new__(cls)
            instance.model_name = model_name
            instance.checkpoint = checkpoint
            instance.model, instance.preprocess, instance.tokenizer = instance._load_model(model_name, checkpoint)
            cls._instances[key] = instance
        return cls._instances[key]
    
    @staticmethod
    def _load_model(model_name: str, checkpoint: str):
        try:
            # Load model
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=model_name, pretrained=checkpoint, cache_dir="./model", device="cuda" if torch.cuda.is_available() else "cpu"
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            return model, preprocess, tokenizer

        except ImportError:
            raise ImportError(
                "Please ensure both open_clip and torch libraries are installed. "
                "pip install open_clip_torch torch"
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        with torch.inference_mode():
                tokens = self.tokenizer(texts).to(self.device)
                text_features = self.model.encode_text(tokens).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
                return text_features.tolist()

    def embed_images(self, uris: list[str]) -> torch.Tensor:

        # Convert base64 strings to PIL images
        images = [Image.open(BytesIO(base64.b64decode(uri))) for uri in uris]

        # Preprocess images and convert to tensors
        processed_images = [self.preprocess(image).unsqueeze(0).to(self.device) for image in images]

        # Concatenate all images into a single batch
        image_tensor = torch.cat(processed_images, dim=0)

        # Generate embeddings
        with torch.inference_mode():
            image_features = self.model.encode_image(image_tensor).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

            return image_features.tolist()