from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())


def draft_email(user_input, name="Nandhini"):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """    
    You are a helpful assistant that drafts an email reply based on an a new email.

    Your goal is to help the user quickly create a perfect email reply.
    
    Keep your reply short and to the point and mimic the style of the email so you reply in a similar manner to match the tone.
    
    Start your reply by saying: "Hi {name}, here's a draft for your reply:". And then proceed with the reply on a new line.
    
    Make sure to sign of with {signature}.
    
    """

    signature = f"Kind regards, \n\{name}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's the email to reply to and consider any other comments from the user for reply as well: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    # response = chain.invoke(user_input=user_input,
    #                         signature=signature, name=name)
    response = chain.invoke(
        {"user_input": user_input, "signature": signature, "name": name})

    return response


def set_reminder(event_name, event_time):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    template = """    
    You are a helpful assistant that sets reminders for calendar events.

    Your goal is to help the user quickly create a reminder for an upcoming event.
    
    Provide the event name and time in a clear and concise manner.
    
    Start your reminder by saying: "Sure! I've set a reminder for {event_name} at {event_time}.". 
    
    And then proceed with any additional details on a new line.
    
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Please set a reminder for the following event: {event_name} at {event_time}."
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.invoke(
        {"event_name": event_name, "event_time": event_time})

    return response


# if __name__ == "__main__":
#     print(draft_email("Hello! I am writing to inquire about the status of my application."))
#     print(set_reminder("Interview", "3:00 PM"))
