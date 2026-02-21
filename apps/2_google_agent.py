from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver


model = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()
memory = MemorySaver()


agent = create_agent(
    model=model,
    tools=[search.run],
    checkpointer=memory,
    system_prompt="You are a agent and can search for any question on google."
)


while True:
    query = input("User: ")
    if query.lower() == "quit":
        print("Good Bye 👋")
        break

    response = agent.invoke(
                {"messages":[{"role":"user", "content":query}]}, 
                {"configurable": {"thread_id": "1"}}, 
            )
    print("AI:",  response["messages"][-1].content)