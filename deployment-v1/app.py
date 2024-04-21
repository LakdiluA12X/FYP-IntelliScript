from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from logging import handlers
# from intelliscript import *

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


temp_qna = "The net interest margin (NIM) in the year 2020 was 3.4%, which was a decline compared to the prior year when it was at 20.2%. This information was reported as at December 2019 and showed a severe deterioration in the asset quality of the banking sector. The sector's net NPL ratio increased to 60% from 20%, and the gross NPL ratio also increased to 74.79% by December 2020 from 20%. These figures suggest that the bank's ability to generate interest income decreased, resulting in a higher level of non-interest expenses. As a result, the net interest margin, which represents the difference between earned interest and total interest expense, decreased from 42.7% at the beginning of 2020 to 3.4% at the end of the same year. This decline indicates that the bank's operational performance and management efficiency were affected by the COVID-19 pandemic and other factors."

temp_report = {
    "type": "report",
    "title": "Central Bank of Sri Lanka (CBSL) 2020 Capital Adequacy Ratio, Net Interest Margin, and Return on Assets Report",
    "sections": [
        {
            "topic": "Introduction",
            "contents": [
              {
                  "type": "paragraph",
                  "content": "The Central Bank of Sri Lanka (CBSL) plays a critical role in maintaining the stability of the Sri Lankan financial system. The financial sector is a vital contributor to the country's economy, with the CBSL responsible for managing the country's interest rates, regulating banks, and supporting economic growth. This report focuses on the capital adequacy ratio (CAR), net interest margin (NIM), and return on assets (ROA) of the CBSL in the year 2020, highlighting the bank's financial performance, challenges, and opportunities."
              },
                {
                  "type": "graph",
                  "content": {
                      "type": "line",
                      "title": "CBSL Financial Performance Trend (2018-2020)",
                      "x_axis_values": [
                          "2018",
                          "2019",
                          "2020"
                      ],
                      "y_axis_values": [
                          "14.5",
                          "15.2",
                          "13.2"
                      ],
                      "x_label": "Year",
                      "y_label": "CAR (%)"
                  }
              }
            ]
        },
        {
            "topic": "CAR in 2020",
            "contents": [
                {
                    "type": "paragraph",
                    "content": "The Central Bank of Sri Lanka (CBSL) plays a critical role in maintaining the stability of the Sri Lankan financial system. The financial sector is a vital contributor to the country's economy, with the CBSL responsible for managing the country's interest rates, regulating banks, and supporting economic growth. This report focuses on the capital adequacy ratio (CAR), net interest margin (NIM), and return on assets (ROA) of the CBSL in the year 2020, highlighting the bank's financial performance, challenges, and opportunities."
                }
            ]
        },
        {
            "topic": "NIM in 2020",
            "contents": [
                {
                    "type": "paragraph",
                    "content": "The net interest margin (NIM) in the year 2020 was 3.4%, which was a decline compared to the prior year when it was at 20.2%. This information was reported as at December 2019 and showed a severe deterioration in the asset quality of the banking sector. The sector's net NPL ratio increased to 60% from 20%, and the gross NPL ratio also increased to 74.79% by December 2020 from 20%. These figures suggest that the bank's ability to generate interest income decreased, resulting in a higher level of non-interest expenses. As a result, the net interest margin, which represents the difference between earned interest and total interest expense, decreased from 42.7% at the beginning of 2020 to 3.4% at the end of the same year. This decline indicates that the bank's operational performance and management efficiency were affected by the COVID-19 pandemic and other factors."
                },
                {
                    "type": "table",
                    "content": {
                        "title": "CBSL Financial Performance Summary (2018-2020)",
                        "header": [
                            "Year",
                            "CAR",
                            "NIM",
                            "ROA"
                        ],
                        "data": [
                            [
                                "2018",
                                "14.5%",
                                "21.3%",
                                "4.1%"
                            ],
                            [
                                "2019",
                                "15.2%",
                                "20.2%",
                                "5.2%"
                            ],
                            [
                                "2020",
                                "13.2%",
                                "3.4%",
                                "3.7%"
                            ]
                        ]
                    }
                }
            ]
        }
    ]
}


temp_insight = {
    "type": "paragraph",
    "content": "The Central Bank of Sri Lanka (CBSL) plays a critical role in maintaining the stability of the Sri Lankan financial system. The financial sector is a vital contributor to the country's economy, with the CBSL responsible for managing the country's interest rates, regulating banks, and supporting economic growth. This report focuses on the capital adequacy ratio (CAR), net interest margin (NIM), and return on assets (ROA) of the CBSL in the year 2020, highlighting the bank's financial performance, challenges, and opportunities."
}


@app.post("/query")
async def user_query(user_query: Query):
    # logger.info(f'User query: {user_query.id}-{user_query.query}')
    # response = response_generation(user_query.query)
    # logger.info(f'LLM response: {user_query.id}-{response}')
    # return {'id': user_query.id, 'content': response}
    return {'id': user_query.id, 'content': temp_qna}


@app.post("/report")
async def user_query(user_query: Query):
    return {'id': user_query.id, 'content': temp_report}


@app.post("/insight")
async def user_query(user_query: Query):
    return {'id': user_query.id, 'content': temp_insight}


@app.post("/feedback")
async def user_query(user_query: Feedback_Query):
    print(user_query.message_id, user_query.mode, user_query.text)


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
