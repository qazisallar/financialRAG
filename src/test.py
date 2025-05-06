from dotenv import load_dotenv
import os
from textwrap import dedent
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_KEY') #INPUT YOUR GROQ KEY
os.environ['PHI_API_KEY'] = os.getenv('AGNO_KEY')  #INPUT YOUR AGNO KEY

print("Groq API key set!" if os.getenv("GROQ_API_KEY") else "Groq API key not set")
print("Agno API key set!" if os.getenv("PHI_API_KEY") else "Phi API key not set")

# Initialize the research agent with advanced journalistic capabilities
research_agent = Agent(
    # model=Groq(id="llama3-70b-8192"),
    model=Groq(id="llama-3.3-70b-versatile"),
    # model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGoTools(), Newspaper4kTools()],
    description=dedent("""\
        You are an elite research analyst in the financial services domain.
        Your expertise encompasses:

        - Deep investigative financial research and analysis
        - fact-checking and source verification
        - Data-driven reporting and visualization
        - Expert interview synthesis
        - Trend analysis and future predictions
        - Complex topic simplification
        - Ethical practices
        - Balanced perspective presentation
        - Global context integration\
    """),
    instructions=dedent("""\
        1. Research Phase
           - Search for 5 authoritative sources on the topic
           - Prioritize recent publications and expert opinions
           - Identify key stakeholders and perspectives

        2. Analysis Phase
           - Extract and verify critical information
           - Cross-reference facts across multiple sources
           - Identify emerging patterns and trends
           - Evaluate conflicting viewpoints

        3. Writing Phase
           - Craft an attention-grabbing headline
           - Structure content in Financial Report style
           - Include relevant quotes and statistics
           - Maintain objectivity and balance
           - Explain complex concepts clearly

        4. Quality Control
           - Verify all facts and attributions
           - Ensure narrative flow and readability
           - Add context where necessary
           - Include future implications
    """),
    expected_output=dedent("""\
        # {Compelling Headline}

        ## Executive Summary
        {Concise overview of key findings and significance}

        ## Background & Context
        {Historical context and importance}
        {Current landscape overview}

        ## Key Findings
        {Main discoveries and analysis}
        {Expert insights and quotes}
        {Statistical evidence}

        ## Impact Analysis
        {Current implications}
        {Stakeholder perspectives}
        {Industry/societal effects}

        ## Future Outlook
        {Emerging trends}
        {Expert predictions}
        {Potential challenges and opportunities}

        ## Expert Insights
        {Notable quotes and analysis from industry leaders}
        {Contrasting viewpoints}

        ## Sources & Methodology
        {List of primary sources with key contributions}
        {website links to the resources used}
        {Research methodology overview}

        ---
        Research conducted by Financial Agent
        Credit Rating Style Report
        Published: {current_date}
        Last Updated: {current_time}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
)

# User Prompt 1
# research_agent.print_response("Analyze the current state and future implications of artificial intelligence in Finance",stream=True,)

# User Prompt 2
research_agent.print_response("Applications of Gen AI in Financial Services",stream=True,)

# User Prompt 3
# research_agent.print_response("AI agentsin Financial Services",stream=True,)