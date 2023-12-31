import json
from openclip import OpenCLIPEmbeddings

def handler(event, context):
    
    if "body" in event:
        event = json.loads(event.get('body'))

    if not ('base64_images' in event or 'texts' in event):
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'No valid data provided in the event'}),
        }
    
    clip_embeddings = OpenCLIPEmbeddings()

    try:
        if 'base64_images' in event:
            base64_images = list(event['base64_images'])
            embeddings = clip_embeddings.embed_base64s(base64_images)
        elif 'texts' in event:
            texts = list(event['texts'])
            embeddings = clip_embeddings.embed_texts(texts)

        return {
            'statusCode': 200,
            'body': json.dumps({'embeddings': embeddings}),
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Error during embedding: {}'.format(e)}),
        }