from dotenv import load_dotenv
import os
from textwrap import dedent
from agno.agent import Agent
from agno.models.groq import Groq
# from agno.tools.duckduckgo import DuckDuckGoTools
# from agno.tools.newspaper4k import Newspaper4kTools
# from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools


load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_KEY') #INPUT YOUR GROQ KEY
os.environ['PHI_API_KEY'] = os.getenv('AGNO_KEY')  #INPUT YOUR AGNO KEY

print("Groq API key set!" if os.getenv("GROQ_API_KEY") else "Groq API key not set")
print("Agno API key set!" if os.getenv("PHI_API_KEY") else "Phi API key not set")


stock_agent = Agent(
    # model=Groq(id="llama3-70b-8192"),
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            historical_prices=True,
            company_info=True,
            company_news=True,
        )
    ],
    instructions=dedent("""\
        You are a seasoned credit rating analyst with deep expertise in market analysis! 📊

        Follow these steps for comprehensive financial analysis:
        1. Market Overview
           - Latest stock price
           - 52-week high and low
        2. Financial Deep Dive
           - Key metrics (P/E, Market Cap, EPS)
        3. Market Context
           - Industry trends and positioning
           - Competitive analysis
           - Market sentiment indicators
           - Analyst Recommendations

        Your reporting style:
        - Begin with an executive summary
        - Use tables for data presentation
        - Include clear section headers
        - Highlight key insights with bullet points
        - Compare metrics to industry averages
        - Include technical term explanations
        - End with a forward-looking analysis

        Risk Disclosure:
        - Always highlight potential risk factors
        - Note market uncertainties
        - Mention relevant regulatory concerns
    """),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
)

print("Stock Agent created. Ready to take user queries..")

# User Query 1
# stock_agent.print_response("What's the latest news and financial performance of NVIDIA Corp (NVDA)?", stream=True)

# User Query 2: Semiconductor market analysis
# stock_agent.print_response(
#     dedent("""\
#     Analyze the semiconductor market performance focusing on:
#     - NVIDIA (NVDA)
#     - AMD (AMD)
#     - Intel (INTC)
#     - Taiwan Semiconductor (TSM)
#     Compare their market positions, growth metrics, and future outlook in terms of AI growth."""),
#     stream=True,
# )

# User Query 3: Competitive analysis
stock_agent.print_response("How is Microsoft performing in the age of AI?", stream=True)