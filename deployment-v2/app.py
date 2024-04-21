from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from logging import handlers
from intelliscript import *
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s|%(message)s')
handler = handlers.TimedRotatingFileHandler(
    "logs/intelliscript.log", when="H", interval=24)
handler.setFormatter(formatter)
logger.addHandler(handler)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    id: int
    query: str


class Feedback_Query(BaseModel):
    message_id: int
    mode: str
    text: str


@app.post("/query")
async def user_query(user_query: Query):
    logger.info(f'User query: {user_query.id}-{user_query.query}')
    response, metadata = qna_response_generator(
        user_query.query, temp=0.85, context_len=1000)
    logger.info(f'LLM response(qna): {user_query.id}-{response}')
    return {'id': user_query.id, 'content': response}


@app.post("/report")
async def user_query(user_query: Query):
    logger.info(f'User query: {user_query.id}-{user_query.query}')
    response = generate_final_report(
        user_query.query, temp=0.85, context_len=2500)
    logger.info(f'LLM response(report): {user_query.id}-{response}')
    with open('test-report.txt', 'w') as outfile:
        outfile.write(str(response))
    return {'id': user_query.id, 'content': response}


@app.post("/insight")
async def user_query(user_query: Query):
    logger.info(f'User query: {user_query.id}-{user_query.query}')
    response, metadata = insight_generator(
        user_query.query, temp=0.85, context_len=1000)
    logger.info(f'LLM response(insight): {user_query.id}-{response}')
    return {'id': user_query.id, 'content': json.loads(response)}


@app.post("/feedback")
async def user_query(user_query: Feedback_Query):
    print(user_query.message_id, user_query.mode, user_query.text)

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
