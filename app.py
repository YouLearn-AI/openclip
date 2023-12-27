import json
from openclip import OpenCLIPEmbeddings
import base64

clip_embeddings = OpenCLIPEmbeddings()

def handler(event, context):

    if not ('images' in event or 'texts' in event):
        return {
            'headers': {'Content-Type': 'application/json'},
            'statusCode': 400,
            'body': json.dumps({'error': 'No valid data provided in the event'}),
        }
    
    try:
        if 'images' in event:
            images = list(event['images'])
            embeddings = clip_embeddings.embed_images(images)
        elif 'texts' in event:
            texts = list(event['texts'])
            embeddings = clip_embeddings.embed_texts(texts)

        return {
            'headers': {'Content-Type': 'application/json'},
            'statusCode': 200,
            'body': json.dumps({'embeddings': embeddings}),
        }
    except Exception as e:
        return {
            'headers': {'Content-Type': 'application/json'},
            'statusCode': 500,
            'body': json.dumps({'error': 'Error during embedding: {}'.format(e)}),
        }
