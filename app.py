import json
from openclip import OpenCLIPEmbeddings
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, context):
    logger.info('Received request with event: {}'.format(event))

    if not ('base64_images' in event or 'texts' in event or 'query' in event):
        logger.error('No valid data provided in the event')
        return {
            'headers': {'Content-Type': 'application/json'},
            'statusCode': 400,
            'body': json.dumps({'error': 'No valid data provided in the event'}),
            'event': event,
        }
    
    clip_embeddings = OpenCLIPEmbeddings()

    try:
        if 'base64_images' in event:
            base64_images = list(event['base64_images'])
            logger.info('Embedding base64 images')
            embeddings = clip_embeddings.embed_base64s(base64_images)
        elif 'texts' in event:
            texts = list(event['texts'])
            logger.info('Embedding documents')
            embeddings = clip_embeddings.embed_documents(texts)
        elif 'query' in event:
            query = str(event['query'])
            logger.info('Embedding query')
            embeddings = clip_embeddings.embed_query(query)

        return {
            'headers': {'Content-Type': 'application/json'},
            'statusCode': 200,
            'body': json.dumps({'embeddings': embeddings}),
            'event': event,
        }
    except Exception as e:
        logger.error('Error during embedding: {}'.format(e))
        return {
            'headers': {'Content-Type': 'application/json'},
            'statusCode': 500,
            'body': json.dumps({'error': 'Error during embedding: {}'.format(e)}),
            'event': event,
        }