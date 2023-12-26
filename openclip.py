from io import BytesIO
import base64
# Ensure that necessary libraries are installed
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

    def __new__(cls, model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k"):
        # Create a unique key based on the model name and checkpoint
        key = (model_name, checkpoint)
        if key not in cls._instances:
            instance = super(OpenCLIPEmbeddings, cls).__new__(cls)
            # Initialize the instance (part of __init__)
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
                model_name=model_name, pretrained=checkpoint
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            return model, preprocess, tokenizer

        except ImportError:
            raise ImportError(
                "Please ensure both open_clip and torch libraries are installed. "
                "pip install open_clip_torch torch"
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        text_features = []
        for text in texts:
            # Tokenize and encode the text
            tokenized_text = self.tokenizer(text)
            embeddings_tensor = self.model.encode_text(tokenized_text)

            # Normalize and convert the embeddings
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)
            embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()
            text_features.append(embeddings_list)

        return text_features

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_image(self, uris: list[str]) -> list[list[float]]:
        # Ensure PIL is available
        if not Image:
            raise ImportError("PIL library not found. Install with 'pip install pillow'")

        image_features = []
        for uri in uris:
            pil_image = Image.open(uri)
            preprocessed_image = self.preprocess(pil_image).unsqueeze(0)
            embeddings_tensor = self.model.encode_image(preprocessed_image)
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)
            embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()
            image_features.append(embeddings_list)

        return image_features

    def embed_base64(self, base64_strings: list[str]) -> list[list[float]]:
        # Ensure PIL is available
        if not Image:
            raise ImportError("PIL library not found. Install with 'pip install pillow'")

        image_features = []
        for base64_str in base64_strings:
            
            image_bytes = base64.b64decode(base64_str)

            image_file = BytesIO(image_bytes)
            image = Image.open(image_file)

            preprocessed_image = self.preprocess(image).unsqueeze(0)
            embeddings_tensor = self.model.encode_image(preprocessed_image)

            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)
            embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()
            image_features.append(embeddings_list)

        return image_features