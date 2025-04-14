# Import necessary libraries
import asyncio
import os

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

import faulthandler
faulthandler.enable()


class Team:
    def __init__(self):

        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API")


        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=openai_api_key,
        )

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
        self.team = MagenticOneGroupChat(
            participants=[search_agent, critic_agent, user_proxy], 
            termination_condition=termination,
            model_client=model_client,
            max_turns=20,
            max_stalls=1
        )

    async def get_team(self):
        await self.team.reset()
        return self.team
