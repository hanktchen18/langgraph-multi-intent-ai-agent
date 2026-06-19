import uuid
import os
import subprocess
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

load_dotenv()

llm = init_chat_model('claude-haiku-4-5-20251001')

KNOWLEDGE = [
    "Alice is a software engineer at Acme Corp, specializing in backend systems and distributed databases.",
    "The Acme Corp headquarters is located in San Francisco, California.",
    "Alice's team is working on Project Orion, a real-time data pipeline that processes over 1 million events per second.",
    "Bob is the product manager for Project Orion and has been at Acme Corp for five years.",
    "The Acme Corp annual hackathon is scheduled for the third week of October.",
    "Alice enjoys hiking and photography in her free time.",
    "Project Orion uses Apache Kafka for message streaming and PostgreSQL for persistent storage.",
]

vector_store = InMemoryVectorStore(GoogleGenerativeAIEmbeddings(model="gemini-embedding-2"))
vector_store.add_documents([Document(page_content=text) for text in KNOWLEDGE])

class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(..., description='Classify whether the user wants to just chat, ask for knowledge or change code in the project.')

# Custom graph state: extends the default with an extra field - message_intent
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None

    next_node: str | None

def classify_intent(state: State):
    # Bind the LLM tor return structured output mathcing IntentClassfier schema
    structured_llm = llm.with_structured_output(IntentClassifier)

    result = structured_llm.invoke([
        {'role': 'system', 'content': ('Classify the user\'s message into one of three intents:\n'
            '- "knowledge": questions about specific people, facts, personal info, '
            'or anything that requires looking up stored information (e.g. "Who is X?", "What does Y do?", "Tell me about Z")\n'
            '- "chat": casual conversation, greetings, opinions, small talk\n'
            '- "code": requests to write, edit, fix, or explain code\n'
            'When in doubt between "chat" and "knowledge", choose "knowledge".')},
        {'role': 'user', 'content': state['messages'][-1].content}
    ])

    # Return the classified intent to update the graph state
    return {'message_intent': result.message_intent}

def accept_coding(state: State):
    user_prompt = state['messages'][-1].content
    decision = interrupt(f'About to run Claude Code with request:\n\n{user_prompt}\n\nApprove? (yes/no, or type a revised request)')

    text = str(decision).strip().lower()

    if text in ['y', 'yes', 'approve', 'ok']:
        return {'next_node': 'coding_agent'}
    
    if text in ['n', 'no', 'deny', 'cancel']:
        return {'messages': [{'role': 'assistant', 'content': 'Coding request was denied by the user.'}], 'next_node': 'denied'}

    return {'messages': [{'role': 'user', 'content': text}], 'next_node': 'accept_coding'}

def prompt_llm_chat(state: State):
    messages = [{'role': 'system', 'content': 'You are a talkative chatbot for fun. Be nice.'}] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]} 

def prompt_llm_rag(state: State):
    query = state['messages'][-1].content
    documents = vector_store.similarity_search(query, k=3)

    context = '\n'.join(f' - {doc.page_content}' for doc in documents)
    
    messages = [{'role': 'system', 'content': f'You are a RAG agent. Answer the user using only the context below. If the answer is not within the context, say I don\'t know and don\'t expose any information of the context. \n\nContext:\n{context}'},] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]} 

def prompt_llm_code(state: State):
    user_prompt = state['messages'][-1].content
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace')

    result = subprocess.run([
        'claude', '-p', user_prompt, '--permission-mode', 'acceptEdits'],
        cwd=workspace,
        capture_output=True,
        text=True
        )

    output = result.stdout.strip() or result.stderr.strip()

    return {'messages': [{'role': 'assistant', 'content': output}]} 

def prepare_coding_request(state: State):
    messages = [
        {'role': 'system', 'content': 'Rewrite the latest user coding request into a clear instruction for Claude Code. Use the conversation history as context. Only output the instruction, no explanation.'},
    ] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]} 

graph_builder = StateGraph(State)

graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('chat_agent', prompt_llm_chat)
graph_builder.add_node('rag_agent', prompt_llm_rag)
graph_builder.add_node('coding_agent', prompt_llm_code)
graph_builder.add_node('prepare_coding_request', prepare_coding_request)
graph_builder.add_node('accept_coding', accept_coding) 

graph_builder.add_edge(START, 'classifier')
graph_builder.add_edge('prepare_coding_request', 'accept_coding')
graph_builder.add_conditional_edges('accept_coding', lambda state: 'end' if state.get('next_node') == 'denied' else state['next_node'], {'end': END, 'coding_agent': 'coding_agent', 'accept_coding': 'prepare_coding_request'})
graph_builder.add_conditional_edges('classifier', lambda state: state['message_intent'], {'chat': 'chat_agent', 'knowledge': 'rag_agent', 'code': 'prepare_coding_request'})

graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('coding_agent', END)


checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

# Each conversation session gets a unique thread ID so state is tracked separately per user/session
config = {'configurable': {'thread_id': uuid.uuid4()}}

while True:
    user_message = input('Enter message: ')
    result = graph.invoke({'messages': [{'role': 'user', 'content': user_message}]}, config=config)

    while '__interrupt__' in result:
        prompt = result['__interrupt__'][0].value
        decision = input(f'{prompt}\n> ')
        result = graph.invoke(Command(resume=decision), config=config)

    print(result['messages'][-1].content)
