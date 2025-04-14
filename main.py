# Import necessary libraries
import asyncio
import os
import yaml

from dotenv import load_dotenv

from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import TextMessage, ToolCallRequestEvent, ToolCallExecutionEvent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.langchain import LangChainToolAdapter
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient

import faulthandler
faulthandler.enable()

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API")


class Agent:
    def __init__(self) -> None:
        # Load the model client from config.
        with open("model_config.yml", "r") as f:
            model_config = yaml.safe_load(f)
        model_client = ChatCompletionClient.load_component(model_config)
        self.search_agent = AssistantAgent(
            name="search",
            model_client=model_client,
            system_message="You are a search agent. You will be given a query and you need to find the information on the web.",
        )
        self.critic_agent = AssistantAgent(
            name="critic",
            model_client=model_client,
            system_message="You are a critic agent. You will be given a plan and you need to critique it. Provide constructive feedback on the plan.",
        )

    async def chat(self, prompt: str) -> str:
        response = await self.search_agent.on_messages(
            [TextMessage(content=prompt, source="user")],
            CancellationToken(),
        )
        assert isinstance(response.chat_message, TextMessage)
        return response.chat_message.content


async def main():
    """
    Main function to initialize and run the travel planning agents.
    """
    # Initialize the OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=openai_api_key,
    )

    # Define system prompts for agents
    search_prompt = (
        """
        You are a search agent.
        You will be given a query and you need to find the information on the web.
        Use the Google search engine to find the information.
        """
    )

    critic_prompt = (
        """
        You are a critic agent.
        You will be given a plan and you need to critique it.
        Provide constructive feedback on the plan.
        Respond with 'APPROVE' when the user is satisfied.
        Don't be very harsh, and APPROVE the plan if it is okay.
        """
    )

    # Initialize the Google Search tool
    search = GoogleSearchAPIWrapper(k=10)
    web_tool = LangChainToolAdapter(
        Tool(
            name="google_search",
            description="Search Google for recent results.",
            func=search.run,
        )
    )

    # Create agents
    search_agent = AssistantAgent(
        name="search",
        model_client=model_client,
        system_message=search_prompt,
        tools=[web_tool],
    )

    critic_agent = AssistantAgent(
        name="critic",
        model_client=model_client,
        system_message=critic_prompt,
    )

    # Define termination condition
    termination = TextMentionTermination("APPROVE")

    # Initialize user proxy agent
    user_proxy = UserProxyAgent(name="user_proxy", input_func=input)

    # Create a team of agents
    team = MagenticOneGroupChat(
        participants=[search_agent, critic_agent, user_proxy], 
        termination_condition=termination,
        model_client=model_client,
        max_turns=20,
        max_stalls=1
    )

    # Run the team and handle messages
    await Console(team.run_stream(task="Plan a cheap trip from Munich"))
        

if __name__ == "__main__":
    asyncio.run(main())