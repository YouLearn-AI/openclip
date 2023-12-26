from io import BytesIO
import base64
import concurrent.futures
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
                model_name=model_name, pretrained=checkpoint
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            return model, preprocess, tokenizer

        except ImportError:
            raise ImportError(
                "Please ensure both open_clip and torch libraries are installed. "
                "pip install open_clip_torch torch"
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        logger.info('Embedding documents')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            text_features = list(executor.map(self._embed_single_text, texts))
        return text_features

    def embed_images(self, uris: list[str]) -> list[list[float]]:
        logger.info('Embedding images')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            image_features = list(executor.map(self._embed_single_image, uris))
        return image_features

    def embed_base64s(self, base64_strings: list[str]) -> list[list[float]]:
        logger.info('Embedding base64s')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            base64_features = list(executor.map(self._embed_single_base64, base64_strings))
        return base64_features

    def _embed_single_text(self, text) -> list[float]:
        logger.info('Embedding single text')
        embeddings_tensor = self._get_embedding_text(text)
        embeddings_list = self._normalize_tensor(embeddings_tensor)
        return embeddings_list
    
    def _embed_single_base64(self, base64_str):
        logger.info('Embedding single base64')
        image_data = self._decode_base64(base64_str)
        embeddings_list = self._embed_single_image(image_data)
        return embeddings_list

    def _embed_single_image(self, image_data):
        logger.info('Embedding single image')
        embeddings_tensor = self._get_embedding_image(image_data)
        embeddings_list = self._normalize_tensor(embeddings_tensor)
        return embeddings_list
    
    def _decode_base64(self, base64_str: str) -> BytesIO:
        logger.info('Decoding base64')
        image_bytes = base64.b64decode(base64_str)
        return BytesIO(image_bytes)
    
    def _get_embedding_text(self, text: str):
        logger.info('Getting text embedding')
        tokenized_text = self.tokenizer(text)
        embeddings_tensor = self.model.encode_text(tokenized_text)
        return embeddings_tensor
    
    def _get_embedding_image(self, image_data: str):
        logger.info('Getting image embedding')
        pil_image = Image.open(image_data)
        preprocessed_image = self.preprocess(pil_image).unsqueeze(0)
        return self.model.encode_image(preprocessed_image)
    
    def _normalize_tensor(self, tensor) -> list[float]:
        logger.info('Normalizing tensor')
        norm = tensor.norm(p=2, dim=1, keepdim=True)
        normalized_tensor = tensor.div(norm)
        return normalized_tensor.squeeze(0).tolist()