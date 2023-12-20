from typing import Union, List

from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain_community.chat_models import ChatOpenAI
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

load_dotenv()


@tool()
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=} ")
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found.")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:
    
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought:
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(temperature=0, stop=["\nObservation"])
    agent = (
        {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()
    )

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {"input": "What is the length in characters of the text DOG?"}
    )
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tools_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tools_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
