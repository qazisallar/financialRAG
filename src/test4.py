from dotenv import load_dotenv
import os
from textwrap import dedent
from agno.agent import Agent
from agno.models.groq import Groq
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

class RAGEvaluator:
    def __init__(self):
        self.evaluator = self._initialize_evaluator()

    def _initialize_evaluator(self):
        return Agent(
            model=Groq(id="llama-3.1-8b-instant"),  # Using different Llama model
            description=dedent("""\
                You are an expert RAG system evaluator with deep expertise in:
                - Information retrieval quality assessment
                - Response accuracy evaluation
                - Source attribution verification
                - Context relevance analysis
                - Natural language generation evaluation
            """),
            instructions=dedent("""\
                Evaluate the RAG system output based on these key metrics:

                1. Faithfulness (1-5):
                   - How accurately does the response reflect the source documents?
                   - Are there any hallucinations or incorrect statements?
                   - Does it maintain factual consistency?

                2. Context Relevance (1-5):
                   - Are the retrieved passages relevant to the query?
                   - Is important context missing?
                   - Is irrelevant information included?

                3. Answer Completeness (1-5):
                   - Does the response fully address the query?
                   - Are all key aspects covered?
                   - Is the level of detail appropriate?

                4. Source Attribution (1-5):
                   - Are sources properly cited?
                   - Is it clear which information comes from where?
                   - Can claims be traced back to sources?

                5. Response Coherence (1-5):
                   - Is the response well-structured?
                   - Does it flow logically?
                   - Is it easy to understand?

                Provide specific examples and explanations for each score.
            """),
            expected_output=dedent("""\
                # RAG Evaluation Report

                ## Overview
                Query: {query}
                Response Length: {n_chars} characters

                ## Metric Scores

                ### Faithfulness: {score}/5
                - Justification:
                - Examples:
                - Areas for Improvement:

                ### Context Relevance: {score}/5
                - Justification:
                - Examples:
                - Areas for Improvement:

                ### Answer Completeness: {score}/5
                - Justification:
                - Examples:
                - Areas for Improvement:

                ### Source Attribution: {score}/5
                - Justification:
                - Examples:
                - Areas for Improvement:

                ### Response Coherence: {score}/5
                - Justification:
                - Examples:
                - Areas for Improvement:

                ## Overall Score: {total}/25

                ## Key Recommendations
                1. {rec1}
                2. {rec2}
                3. {rec3}

                ## Summary
                {final_assessment}
            """),
            markdown=True,
        )

    def evaluate(self, query: str, response: str, context: list, stream: bool = True):
        """
        Evaluate a RAG system's response

        Args:
            query (str): Original user query
            response (str): RAG system's response
            context (list): Retrieved passages used for the response
            stream (bool): Whether to stream the evaluation output
        """
        evaluation_prompt = f"""
        Please evaluate this RAG system output:

        QUERY:
        {query}

        RETRIEVED CONTEXT:
        {' '.join(context)}

        RESPONSE:
        {response}

        Provide a detailed evaluation following the metrics and format specified.
        """

        return self.evaluator.print_response(evaluation_prompt, stream=stream)


# Initialize evaluator
evaluator = RAGEvaluator()
print("LLM-as-a Judge Evaluator initialized successfully!")

# Example evaluation. Rerun this to use actual financial RAG outputs
# query = "What are the key features of transformer models?"
# context = [
#     "Transformer models use self-attention mechanisms to process input sequences.",
#     "Key features include parallel processing and handling of long-range dependencies."
# ]
# response = "Transformer models are characterized by their self-attention mechanism..."


# Example query, response, and context
# User Query 3: Competitive analysis
query = "How is Microsoft performing in the age of AI?"
context = [
    "Microsoft has been investing heavily in AI technologies, including partnerships with OpenAI.",
    "The company's Azure cloud platform is a key driver of its AI strategy."
]
response = stock_agent.print_response(query, stream=False)

# Run evaluation
evaluator.evaluate(query, response, context)