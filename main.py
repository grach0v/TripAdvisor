from dotenv import load_dotenv
import os
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, ToolCallRequestEvent, ToolCallExecutionEvent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API")

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=openai_api_key,
    )

    planner_prompt = """
    You are a travel planner. 
    You will be given a user query and you need to plan a trip for the user.
    Yoo should create a plan of the trip that will be passed to the other agents to find information in the web.
    Give clear instructions what information should be googled.
    """

    search_prompt = """
    You are a search agent.
    You will be given a query and you need to find the information in the web.
    You should use the Google search engine to find the information.
    """

    critic_prompt = """
    You are a critic agent.
    You will be given a plan and you need to critique it.
    You should provide constructive feedback on the plan.
    Respond with 'APPROVE' to when user is satisfied.
    """

    search = GoogleSearchAPIWrapper(k=10)

    web_tool = LangChainToolAdapter(
        Tool(
            name="google_search",
            description="Search Google for recent results.",
            func=search.run,
        )
    )

    planner_agent = AssistantAgent(
        "planner",
        model_client=model_client,
        system_message=planner_prompt,
    )

    search_agent = AssistantAgent(
        "search",
        model_client=model_client,
        system_message=search_prompt,
        tools=[web_tool],
    )

    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message=critic_prompt,
    )

    termination = TextMentionTermination("APPROVE")

    user_proxy = UserProxyAgent("user_proxy", input_func=input)

    team = MagenticOneGroupChat(
        [planner_agent, search_agent, critic_agent, user_proxy], 
        termination_condition=termination,
        model_client=model_client,
        max_turns=20,
        max_stalls=1
    )

    async for message in team.run_stream(task="Plan a cheap trip from Munich"): 
        if isinstance(message, TextMessage):
            print("Agent:", message.source)
            print("Message:", message.content)
            print("\n\n")
        
        elif isinstance(message, ToolCallRequestEvent):
            print("ToolCallRequestEvent Agent:", message.source)
            for content in message.content:
                print("Message:", content)
            
            print("\n\n")

        elif isinstance(message, ToolCallExecutionEvent):
            print("ToolCallExecutionEvent Agent:", message.source)
            for content in message.content:
                print("Message:", content)
            

        elif isinstance(message, TaskResult):
            print("Stop Reason:", message.stop_reason)

        else:
            print(message)
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())