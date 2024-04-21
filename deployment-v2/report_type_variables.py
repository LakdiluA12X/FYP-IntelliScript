import random
import json

Financial_System_Stability = ['capital adequacy ratio',
                              'net interest margin',
                              'return on assets',
                              'return on equity', 'banking sector in Sri Lanka',
                              'Household and Corporate Sector Analysis',
                              'Sustainable Financing',
                              'Financial Stability Indicators and Maps',
                              'Solvency Stress Testing',
                              'Liquidity Stress Testing',
                              'Systematic Risk Survey',
                              'Periodical risk assessments',
                              'Microfinance',
                              'Money Broking',
                              'Licensed Finance Companies',
                              'Specialised Leasing Companies',
                              'Financial Market Infrastructure']
acts_laws = ["Central Bank of Sri Lanka Act",
             "Monetary Law Act",
             "Banking (Special Provisions) Act",
             "Banking Act",
             "Finance Business Act",
             "Finance Leasing Act",
             "Micro finance Act",
             "Foreign Exchange Act",
             "Exchange Control Act",
             "Employees Provident Fund (Amendment) Act",
             "Financial Transactions Reporting Act",
             "Active Liability Management Act"
             ]

Monetary_Policy_Reports = ["interest rates",
                           "policy rates",
                           "inflation targeting",
                           "money supply",
                           "liquidity management",
                           "reserve requirements",
                           ]

Economic_Performance_Reports = ["GDP growth",
                                "economic indicators",
                                "sector performance",
                                "industrial output",
                                "services output",
                                "economic outlook",
                                ]

Inflation_and_Price_Index_Reports = ["Consumer Price Index (CPI)",
                                     "Wholesale Price Index (WPI)",
                                     "inflation rate",
                                     "core inflation",
                                     "price trends",
                                     ]

Exchange_Rate_Reports = ["exchange rate fluctuations",
                         "currency trends",
                         "foreign exchange markets",
                         "currency valuation",
                         ]

Balance_of_Payments_Reports = ["trade balance",
                               "current account",
                               "capital account",
                               "foreign direct investment",
                               "net exports",
                               "external sector",
                               ]

Financial_Sector_Reports = ["banking sector",
                            "financial institutions",
                            "asset quality",
                            "non-performing loans",
                            "capital adequacy",
                            ]

Government_Finance_Reports = ["government revenue",
                              "public expenditure",
                              "budget deficit",
                              "public debt",
                              "fiscal policy",
                              ]

Interest_Rate_Reports = ["interest rate trends",
                         "commercial rates",
                         "deposit rates",
                         "lending rates",
                         "yield curve"
                         ]

Banking_and_Financial_Institution_Reports = ["liquidity",
                                             "solvency",
                                             "profitability",
                                             "capital adequacy",
                                             "non-banking financial institutions",
                                             ]

Investment_and_Savings_Reports = ["domestic investment",
                                  "foreign investment",
                                  "household savings",
                                  "corporate savings",
                                  "investment climate",
                                  ]

Labor_Market_Reports = ["employment rate",
                        "unemployment rate",
                        "labor force participation",
                        "wage growth",
                        "labor market dynamics",
                        ]

Trade_Reports = ["import trends",
                 "export trends",
                 "trade balances",
                 "major trading partners",
                 "commodity trade",
                 ]

Poverty_and_Income_Distribution_Reports = ["poverty rate",
                                           "income inequality",
                                           "Gini coefficient",
                                           "poverty alleviation programs",
                                           ]

Economic_Outlook_Reports = ["economic forecasts",
                            "future trends",
                            "economic risks",
                            "scenario analysis",
                            ]

lists_dict = {
    "Financial_System_Stability": Financial_System_Stability,
    "acts_laws": acts_laws,
    "Monetary_Policy_Reports": Monetary_Policy_Reports,
    "Economic_Performance_Reports": Economic_Performance_Reports,
    "Inflation_and_Price_Index_Reports": Inflation_and_Price_Index_Reports,
    "Exchange_Rate_Reports": Exchange_Rate_Reports,
    "Balance_of_Payments_Reports": Balance_of_Payments_Reports,
    "Financial_Sector_Reports": Financial_Sector_Reports,
    "Government_Finance_Reports": Government_Finance_Reports,
    "Interest_Rate_Reports": Interest_Rate_Reports,
    "Banking_and_Financial_Institution_Reports": Banking_and_Financial_Institution_Reports,
    "Investment_and_Savings_Reports": Investment_and_Savings_Reports,
    "Labor_Market_Reports": Labor_Market_Reports,
    "Trade_Reports": Trade_Reports,
    "Poverty_and_Income_Distribution_Reports": Poverty_and_Income_Distribution_Reports,
    "Economic_Outlook_Reports": Economic_Outlook_Reports,
}


empty_report_temp = {
    "type": "report",
    "title": "I couldn't find enough information to generate the requested report. Please try a different report type.",
    "sections": []
}

empty_report = json.dumps(empty_report_temp)


def get_financial_report_topics(query):
    keywords_to_category = {
        "financial stability": "Financial_System_Stability",
        "stability": "Financial_System_Stability",
        "acts": "acts_laws",
        "laws": "acts_laws",
        "monetary policy": "Monetary_Policy_Reports",
        "interest rates": "Monetary_Policy_Reports",
        "money supply": "Monetary_Policy_Reports",
        "economic performance": "Economic_Performance_Reports",
        "GDP growth": "Economic_Performance_Reports",
        "sector performance": "Economic_Performance_Reports",
        "inflation": "Inflation_and_Price_Index_Reports",
        "CPI": "Inflation_and_Price_Index_Reports",
        "WPI": "Inflation_and_Price_Index_Reports",
        "exchange rate": "Exchange_Rate_Reports",
        "currency": "Exchange_Rate_Reports",
        "trade balance": "Balance_of_Payments_Reports",
        "current account": "Balance_of_Payments_Reports",
        "foreign direct investment": "Balance_of_Payments_Reports",
        "financial institutions": "Financial_Sector_Reports",
        "government revenue": "Government_Finance_Reports",
        "public expenditure": "Government_Finance_Reports",
        "budget deficit": "Government_Finance_Reports",
        "fiscal policy": "Government_Finance_Reports",
        "interest rate": "Interest_Rate_Reports",
        "commercial rates": "Interest_Rate_Reports",
        "deposit rates": "Interest_Rate_Reports",
        "lending rates": "Interest_Rate_Reports",
        "yield curve": "Interest_Rate_Reports",
        "liquidity": "Banking_and_Financial_Institution_Reports",
        "solvency": "Banking_and_Financial_Institution_Reports",
        "profitability": "Banking_and_Financial_Institution_Reports",
        "capital adequacy": "Banking_and_Financial_Institution_Reports",
        "investment and savings": "Investment_and_Savings_Reports",
        "domestic investment": "Investment_and_Savings_Reports",
        "foreign investment": "Investment_and_Savings_Reports",
        "household savings": "Investment_and_Savings_Reports",
        "corporate savings": "Investment_and_Savings_Reports",
        "investment climate": "Investment_and_Savings_Reports",
        "labor market": "Labor_Market_Reports",
        "employment": "Labor_Market_Reports",
        "unemployment": "Labor_Market_Reports",
        "wage growth": "Labor_Market_Reports",
        "labor force participation": "Labor_Market_Reports",
        "import trends": "Trade_Reports",
        "export trends": "Trade_Reports",
        "major trading partners": "Trade_Reports",
        "poverty": "Poverty_and_Income_Distribution_Reports",
        "income inequality": "Poverty_and_Income_Distribution_Reports",
        "Gini coefficient": "Poverty_and_Income_Distribution_Reports",
        "economic forecasts": "Economic_Outlook_Reports",
        "future trends": "Economic_Outlook_Reports",
        "economic risks": "Economic_Outlook_Reports"
    }

    for keyword, category in keywords_to_category.items():
        if keyword in query.lower():
            if category in lists_dict:
                if len(lists_dict[category]) > 3:
                    selected_topics = []
                    while len(selected_topics) < 3:
                        topic = random.choice(lists_dict[category])
                        if topic not in selected_topics:
                            selected_topics.append(topic)
                    return category, selected_topics

                else:
                    return category, lists_dict[category]

    return "No matching category found.", []
