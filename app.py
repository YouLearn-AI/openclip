import json
from openclip import OpenCLIPEmbeddings

def handler(event, context):

    event_payload = event['queryStringParameters']
    if not ('base64_images' in event_payload or 'texts' in event_payload):
        return {
            'headers': {'Content-Type': 'application/json'},
            'statusCode': 400,
            'body': json.dumps({'error': 'No valid data provided in the event'}),
        }
    
    clip_embeddings = OpenCLIPEmbeddings()

    try:
        if 'base64_images' in event_payload:
            base64_images = list(event_payload['base64_images'])
            embeddings = clip_embeddings.embed_base64s(base64_images)
        elif 'texts' in event_payload:
            texts = list(event_payload['texts'])
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