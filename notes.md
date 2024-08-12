# LangGraph

## LangGraph boilerplate output objects at every stage

#### here is tool creation so a funciton that is having the decorator `@tool` (`from langchain_core.tools import tool`)
```python 
tools = [search]
# Outputs
Tools:  [StructuredTool(name='search', description='Call to surf the web.', args_schema=<class 'pydantic.v1.main.searchSchema'>, func=<function search at 0x7a18160244c0>)]
```
#### creation of a node type tool
```python
tool_node = ToolNode(tools)
# outputs
tool_node:  tools(recurse=True, tools_by_name={'search': StructuredTool(name='search', description='Call to surf the web.', args_schema=<class 'pydantic.v1.main.searchSchema'>, func=<function search at 0x7a18160244c0>)}, handle_tool_errors=True)
```
#### creatiion of model adding the `bind_tool` and passing the tool in it in order to have the Openai tool mimmicing done and not get the error `missing bind` or something like that 
```python
model = groq_llm_llama3_70b.bind_tools(tools)
# outputs
Model with bind_tools:  bound=ChatGroq(client=<groq.resources.chat.completions.Completions object at 0x7a1816042680>, async_client=<groq.resources.chat.completions.AsyncCompletions object at 0x7a18160419c0>, model_name='llama3-70b-8192', temperature=0.1, groq_api_key=SecretStr('**********'), max_tokens=1024) kwargs={'tools': [{'type': 'function', 'function': {'name': 'search', 'description': 'Call to surf the web.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}}]}
```
#### defining a new graph using state to track messages
```python
workflow = StateGraph(MessagesState)
# outputs
workflow:  <langgraph.graph.state.StateGraph object at 0x7a1816ebd150>
```
#### adding different node to the graph
```python
workflow.add_node("agent", call_model)
# outputs
Workflow add node 'agent':  <langgraph.graph.state.StateGraph object at 0x7a1816ebd150>

workflow.add_node("tools", tool_node)
# outputs
Workflow add node 'tools':  <langgraph.graph.state.StateGraph object at 0x7a1816ebd150>
```
#### here we set where the graph will start `set_entrypoint`
```python
workflow.set_entry_point("agent")
# outputs
Workflow set entry point 'agent':  <langgraph.graph.state.StateGraph object at 0x7a1816ebd150>
```
#### here we define the lines of the workflow with `add_conditional_edges`, the function being the one determining the next step in function of its output
```python
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)
# outputs
Workflow add conditional edge 'agent' -> should_continue func:  <langgraph.graph.state.StateGraph object at 0x7a1816ebd150>
```
#### here we define the lines of the workflow with a 'normal edge' this time from `tools` to `agent`
```python
workflow.add_edge("tools", 'agent')
# outputs
Workflow add edge 'tools' -> 'agent':  <langgraph.graph.state.StateGraph object at 0x7a1816ebd150>
```
#### here we set up a checkpointer that persistent state between runs like a memory `from langgraph.checkpoint import MemorySaver`. (we might study this function more thouroughly to see if we can implement custom functions for other purposes using this method of checkpointing)
```python
checkpointer = MemorySaver()
# outputs
Memory checkpointer:  <langgraph.checkpoint.memory.MemorySaver object at 0x7a18160323b0>
```
#### here we compile the graph to become a LangChain runnable therefore we will be able to use .invoke(), .stream(), .batch(), and the memory checkpointer here is optional
```python
app = workflow.compile(checkpointer=checkpointer)
# outputs
App compiled with checkpointer:  nodes={'__start__': PregelNode(config={'tags': ['langsmith:hidden'], 'metadata': {}, 'configurable': {}}, channels=['__start__'], triggers=['__start__'], writers=[ChannelWrite<messages>(recurse=True, writes=[ChannelWriteEntry(channel='messages', value=<object object at 0x7a181858b370>, skip_none=False, mapper=_get_state_key(recurse=False))], require_at_least_one_of=['messages']), ChannelWrite<start:agent>(recurse=True, writes=[ChannelWriteEntry(channel='start:agent', value='__start__', skip_none=False, mapper=None)], require_at_least_one_of=None)]), 'agent': PregelNode(config={'tags': [], 'metadata': {}, 'configurable': {}}, channels={'messages': 'messages'}, triggers=['tools', 'start:agent'], mapper=functools.partial(<function _coerce_state at 0x7a181657a830>, <class 'langgraph.graph.message.MessagesState'>), writers=[ChannelWrite<agent,messages>(recurse=True, writes=[ChannelWriteEntry(channel='agent', value='agent', skip_none=False, mapper=None), ChannelWriteEntry(channel='messages', value=<object object at 0x7a181858b370>, skip_none=False, mapper=_get_state_key(recurse=False))], require_at_least_one_of=['messages']), _route(recurse=True, _is_channel_writer=True)]), 'tools': PregelNode(config={'tags': [], 'metadata': {}, 'configurable': {}}, channels={'messages': 'messages'}, triggers=['branch:agent:should_continue:tools'], mapper=functools.partial(<function _coerce_state at 0x7a181657a830>, <class 'langgraph.graph.message.MessagesState'>), writers=[ChannelWrite<tools,messages>(recurse=True, writes=[ChannelWriteEntry(channel='tools', value='tools', skip_none=False, mapper=None), ChannelWriteEntry(channel='messages', value=<object object at 0x7a181858b370>, skip_none=False, mapper=_get_state_key(recurse=False))], require_at_least_one_of=['messages'])])} channels={'messages': <langgraph.channels.binop.BinaryOperatorAggregate object at 0x7a1816032410>, '__start__': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7a1816031510>, 'agent': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7a18160312a0>, 'tools': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7a1816031120>, 'start:agent': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7a18160313c0>, 'branch:agent:should_continue:tools': <langgraph.channels.ephemeral_value.EphemeralValue object at 0x7a18160311e0>} auto_validate=False stream_mode='updates' output_channels=['messages'] stream_channels=['messages'] input_channels='__start__' checkpointer=<langgraph.checkpoint.memory.MemorySaver object at 0x7a18160323b0> builder=<langgraph.graph.state.StateGraph object at 0x7a1816ebd150>
```
#### here we use the Langchain Graph runnable created and use .invoke(), the `{"thread_id": 42}` can be used again to use the memory in order to get same kind of response but adapted to the new user query. so if same ID is used we will get same kind of response formatting.
```python
final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": 42}}
)
# outputs
messages from call_model func:  [HumanMessage(content='what is the weather in sf', id='75ade55f-d939-46d5-926e-fa293af0cf14')]
```
#### here we see the function called logs. Here for 
```python
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    print("messages from should_continue func: ", messages)
    last_message = messages[-1]
    print("last message from should_continue func: ", last_message)
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        print("Tool called!")
        return "tools"
    # Otherwise, we stop (reply to the user)
    print("Tool not called returning answer to user.")
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    print("messages from call_model func: ", messages)
    response = model.invoke(messages)
    print("response from should_continue func: ", response)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


## flow of outputs when 'final_state' invokes the Graph runnable 'app'
## agent > conditional edge > tool call > tools edge answer to agent edge > answer to user 
##                          > tool not called > answer to user
# Here teh conditional edge decides if tool needs to be called and decides to call the tool to get an answer
response from should_continue func:  content='' additional_kwargs={'tool_calls': [{'id': 'call_4gr6', 'function': {'arguments': '{"query":"weather in san francisco"}', 'name': 'search'}, 'type': 'function'}]} response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 903, 'total_tokens': 949, 'completion_time': 0.14465532, 'prompt_time': 0.065623244, 'queue_time': None, 'total_time': 0.210278564}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-09d0c91f-a6c9-476d-bc6e-cb332d4757b2-0' tool_calls=[{'name': 'search', 'args': {'query': 'weather in san francisco'}, 'id': 'call_4gr6', 'type': 'tool_call'}] usage_metadata={'input_tokens': 903, 'output_tokens': 46, 'total_tokens': 949}
messages from should_continue func:  [HumanMessage(content='what is the weather in sf', id='75ade55f-d939-46d5-926e-fa293af0cf14'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_4gr6', 'function': {'arguments': '{"query":"weather in san francisco"}', 'name': 'search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 903, 'total_tokens': 949, 'completion_time': 0.14465532, 'prompt_time': 0.065623244, 'queue_time': None, 'total_time': 0.210278564}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-09d0c91f-a6c9-476d-bc6e-cb332d4757b2-0', tool_calls=[{'name': 'search', 'args': {'query': 'weather in san francisco'}, 'id': 'call_4gr6', 'type': 'tool_call'}], usage_metadata={'input_tokens': 903, 'output_tokens': 46, 'total_tokens': 949})]
last message from should_continue func:  content='' additional_kwargs={'tool_calls': [{'id': 'call_4gr6', 'function': {'arguments': '{"query":"weather in san francisco"}', 'name': 'search'}, 'type': 'function'}]} response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 903, 'total_tokens': 949, 'completion_time': 0.14465532, 'prompt_time': 0.065623244, 'queue_time': None, 'total_time': 0.210278564}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-09d0c91f-a6c9-476d-bc6e-cb332d4757b2-0' tool_calls=[{'name': 'search', 'args': {'query': 'weather in san francisco'}, 'id': 'call_4gr6', 'type': 'tool_call'}] usage_metadata={'input_tokens': 903, 'output_tokens': 46, 'total_tokens': 949}
Tool called!
# here the model is called to get an answer
messages from call_model func:  [HumanMessage(content='what is the weather in sf', id='75ade55f-d939-46d5-926e-fa293af0cf14'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_4gr6', 'function': {'arguments': '{"query":"weather in san francisco"}', 'name': 'search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 903, 'total_tokens': 949, 'completion_time': 0.14465532, 'prompt_time': 0.065623244, 'queue_time': None, 'total_time': 0.210278564}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-09d0c91f-a6c9-476d-bc6e-cb332d4757b2-0', tool_calls=[{'name': 'search', 'args': {'query': 'weather in san francisco'}, 'id': 'call_4gr6', 'type': 'tool_call'}], usage_metadata={'input_tokens': 903, 'output_tokens': 46, 'total_tokens': 949}), ToolMessage(content="It's 60 degrees and foggy.", name='search', id='f60bbd34-9c08-48ad-b6bb-74c96b741a31', tool_call_id='call_4gr6')]
# here tool is not called and returning answer to user. This because we called it twice with same user query and the second time here it uses the memory to form its answer.
response from should_continue func:  content='The weather in San Francisco is 60 degrees and foggy.' response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 979, 'total_tokens': 993, 'completion_time': 0.042933596, 'prompt_time': 0.074989348, 'queue_time': None, 'total_time': 0.117922944}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'stop', 'logprobs': None} id='run-576d3c0f-6915-4fd3-858d-2532c76ed5c0-0' usage_metadata={'input_tokens': 979, 'output_tokens': 14, 'total_tokens': 993}
messages from should_continue func:  [HumanMessage(content='what is the weather in sf', id='75ade55f-d939-46d5-926e-fa293af0cf14'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_4gr6', 'function': {'arguments': '{"query":"weather in san francisco"}', 'name': 'search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 903, 'total_tokens': 949, 'completion_time': 0.14465532, 'prompt_time': 0.065623244, 'queue_time': None, 'total_time': 0.210278564}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-09d0c91f-a6c9-476d-bc6e-cb332d4757b2-0', tool_calls=[{'name': 'search', 'args': {'query': 'weather in san francisco'}, 'id': 'call_4gr6', 'type': 'tool_call'}], usage_metadata={'input_tokens': 903, 'output_tokens': 46, 'total_tokens': 949}), ToolMessage(content="It's 60 degrees and foggy.", name='search', id='f60bbd34-9c08-48ad-b6bb-74c96b741a31', tool_call_id='call_4gr6'), AIMessage(content='The weather in San Francisco is 60 degrees and foggy.', response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 979, 'total_tokens': 993, 'completion_time': 0.042933596, 'prompt_time': 0.074989348, 'queue_time': None, 'total_time': 0.117922944}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'stop', 'logprobs': None}, id='run-576d3c0f-6915-4fd3-858d-2532c76ed5c0-0', usage_metadata={'input_tokens': 979, 'output_tokens': 14, 'total_tokens': 993})]
last message from should_continue func:  content='The weather in San Francisco is 60 degrees and foggy.' response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 979, 'total_tokens': 993, 'completion_time': 0.042933596, 'prompt_time': 0.074989348, 'queue_time': None, 'total_time': 0.117922944}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'stop', 'logprobs': None} id='run-576d3c0f-6915-4fd3-858d-2532c76ed5c0-0' usage_metadata={'input_tokens': 979, 'output_tokens': 14, 'total_tokens': 993}
Tool not called returning answer to user.

# if we want the full object with all answers
# first call for answer with tool used and stored to memory
final_state
# outputs 
Final State = answer:  {'messages': [HumanMessage(content='what is the weather in sf', id='75ade55f-d939-46d5-926e-fa293af0cf14'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_4gr6', 'function': {'arguments': '{"query":"weather in san francisco"}', 'name': 'search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 46, 'prompt_tokens': 903, 'total_tokens': 949, 'completion_time': 0.14465532, 'prompt_time': 0.065623244, 'queue_time': None, 'total_time': 0.210278564}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-09d0c91f-a6c9-476d-bc6e-cb332d4757b2-0', tool_calls=[{'name': 'search', 'args': {'query': 'weather in san francisco'}, 'id': 'call_4gr6', 'type': 'tool_call'}], usage_metadata={'input_tokens': 903, 'output_tokens': 46, 'total_tokens': 949}), ToolMessage(content="It's 60 degrees and foggy.", name='search', id='f60bbd34-9c08-48ad-b6bb-74c96b741a31', tool_call_id='call_4gr6'), AIMessage(content='The weather in San Francisco is 60 degrees and foggy.', response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 979, 'total_tokens': 993, 'completion_time': 0.042933596, 'prompt_time': 0.074989348, 'queue_time': None, 'total_time': 0.117922944}, 'model_name': 'llama3-70b-8192', 'system_fingerprint': 'fp_c1a4bcec29', 'finish_reason': 'stop', 'logprobs': None}, id='run-576d3c0f-6915-4fd3-858d-2532c76ed5c0-0', usage_metadata={'input_tokens': 979, 'output_tokens': 14, 'total_tokens': 993})]}

# if we want the last message only from the stte which is the answer:
# second call for answer no need tool call, using memory to form it's answer to same query
final_state["messages"][-1].content
# outputs
Final state last message content:  The weather in San Francisco is 60 degrees and foggy.
```

## LangGraph Object needed or interesting information to get from objects

##### LLM Response last message content (add [-1] to target last message if needed only that one)
reponse["messages"][-1].content

##### AI LLM Response token usage data
response["response_metadata"]["token_usage"]
response["response_metadata"]["completion_tokens"]
response["response_metadata"]["prompt_tokens"]
response["response_metadata"]["total_tokens"]
response["response_metadata"]["completion_time"]
response["response_metadata"]["prompt_time"]
response["response_metadata"]["total_time"]
response["model_name"]
response["usage_metadata"]["input_tokens"]
response["usage_metadata"]["output_tokens"]
response["usage_metadata"]["total_tokens"]

if tool called we bet this additional part inside of it:
response["aDDitional-kwargs"]["tool_calls"][0]["function"]["arguments"]["query"] # argument passed in the function
response["aDDitional-kwargs"]["tool_calls"][0]["function"]["name"] # name of function
response["aDDitional-kwargs"]["tool_calls"][0]["type"] # type function for example
                                         additional_kwargs={
                                           'tool_calls': [
                                             {
                                               'id': 'call_p5pe',
                                               'function': {
                                                 'arguments': '{
                                                   "query":"current weather in san francisco"
                                                 }',
                                                 'name': 'search'
                                               },
                                               'type': 'function'
                                             }
                                           ]
                                         }

##### Tool Message Object (its content and function name)
response.content
response.name 

##### Human Message object (its content only)
response.content

##### function call LLM list structure
[Human message, AI message, Tool message]


# LANGCHAIN & LANGGRAPH PRECISIONS ON WORKFLOWS

## PROMPTING
- PROMPT = prompt_template, input_vars=[...,...,...]
- type_of_action = PROMPT  |  LLM  |  PARSER(if any)

- eg.:
input_var1 = user_query
input_var2 = func(....) output

response = type_of_action.invoke(
  {
    "input_var1": input_var1 ,
    "input_var2": input_var2
  }
)


## GRAPH LOGIC
- All possible nodes:
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("web_search", web_search)  # web search

- All nodes used to make workflow, start route flow:
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

- Like a 'IF/ELSE' which will route depending on output of function 
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "search": "web_search", # if output == "search" use web_search node
        "generate": "generate",  # if output == "generate" use generate node
    },
)

- After 'IF/ELSE' possible routes and the next logic
workflow.add_edge("web_search", "generate") # OR web_search, then Generate and then END
workflow.add_edge("generate", END) # OR generate, then END

## Biriculouskous Diagram of the worlfow
--- START --- Retrieve --- Grade_documents (uses decide_to_generate func output to decide where to go after) --- output = search --- Web_search --- Generate --- END
                                                                                                             --- output = generate --- Generate --- END
#### Diagram
[Diagram Link:](https://excalidraw.com/#json=Id5LJIJ2GnPsu3Fk6E1Nr,39wVy9-u6aX7PJY-YRPirw)

[---------------------------------------------------------]

# LangGraph States
- Use of Pydantic to have multi-states enabled in LangGraph
### define states
from pydantic import BaseModel
from typing import List

class MessageState(BaseModel):
    messages: List[str]

class UserState(BaseModel):
    user_id: int
    preferences: dict

### initialize
message_state = MessageState(messages=[])
user_state = UserState(user_id=1, preferences={})

### Node access to states
def update_message_state(state: MessageState, new_message: str):
    state.messages.append(new_message)
    return state

def update_user_state(state: UserState, new_preferences: dict):
    state.preferences.update(new_preferences)
    return state

### Example of use
from langgraph import StateGraph

graph = StateGraph()

graph.add_node("update_message_state", update_message_state)
graph.add_node("update_user_state", update_user_state)

graph.add_edge("start", "update_message_state")
graph.add_edge("update_message_state", "update_user_state")

- OR Use Composite state class
class CompositeState(BaseModel):
    message_state: MessageState
    user_state: UserState

composite_state = CompositeState(
    message_state=MessageState(messages=[]),
    user_state=UserState(user_id=1, preferences={})
)

### Nodes
def update_message_in_composite_state(state: CompositeState, new_message: str):
    state.message_state.messages.append(new_message)
    return state

def update_user_in_composite_state(state: CompositeState, new_preferences: dict):
    state.user_state.preferences.update(new_preferences)
    return state

### Graph
graph = StateGraph()

graph.add_node("update_message", update_message_in_composite_state)
graph.add_node("update_user", update_user_in_composite_state)

graph.add_edge("start", "update_message")
graph.add_edge("update_message", "update_user")



# LangGraph States & Langfuse Tracing

See also if we can use langfuse prompts in order to have prompts injected in the code:
eg.:
### Build and create the prompt
langfuse.create_prompt(
    name="movie-critic",
    type="chat", # can be "text" for chat prompt
    prompt="Do you like {{movie}}?",
    labels=["production"],  # directly promote to production
    config={
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "supported_languages": ["en", "fr"],
    },  # optionally, add configs (e.g. model parameters or model tools) or tags
)

from langfuse import Langfuse
###### Initialize Langfuse client
langfuse = Langfuse()
###### Get current production version of a text prompt
prompt = langfuse.get_prompt("movie-critic")
###### Insert variables into prompt template
compiled_prompt = prompt.compile(movie="Dune 2")
###### "Do you like Dune 2?"

###### Get current production version of a chat prompt
chat_prompt = langfuse.get_prompt("movie-critic-chat", type="chat") # type arg infers the prompt type (default is 'text') 
###### Insert variables into chat prompt template
compiled_chat_prompt = chat_prompt.compile(movie="Dune 2")
###### [
     {
       "role": "system",
       "content": "You are an expert on Dune 2"
     }
   ]
- some commands for the prompts
###### Get specific version
prompt = langfuse.get_prompt("movie-critic", version=1)
###### Get specific label
prompt = langfuse.get_prompt("movie-critic", label="staging")
###### Get latest prompt version. The 'latest' label is automatically maintained by Langfuse.
prompt = langfuse.get_prompt("movie-critic", label="latest")
###### Extend cache TTL from default 60 to 300 seconds
prompt = langfuse.get_prompt("movie-critic", cache_ttl_seconds=300)

- access RAW prompt and configs
###### Raw prompt including {{variables}}. For chat prompts, this is a list of chat messages.
prompt.prompt
###### Config object
prompt.config

- Langfuse tracing in function:
@observe(as_type="generation")
def nested_generation():
    prompt = langfuse.get_prompt("movie-critic")
 
    langfuse_context.update_current_observation(
        prompt=prompt,
    )
    
#### The need of reducer function change the behavior of LangGraph States
LangGraph state will update at every node getting rid of the previous value, therefore, if you want to update but keep track of changes you need a reducer function for that.
That is when `Annotated` from `tying` is coming to permit to have a reducer added to your state, then you would just use that function to for example add values to your state.
```python
from typing import TypedDict, Annotated
from operator import add
class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]

# OR here
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

#### If you don't want to use reducer, you can use prebuilt method that adds up messages to your state like ``
from langgraph.graph import MessagesState
class State(MessagesState):
    documents: list[str]
[------------------------------------------------------------------------------------------------------]


# LangGraph Edges
A node can have MULTIPLE outgoing edges. If a node has multiple out-going edges, all of those destination nodes will be executed in parallel as a part of the next superstep.
node --- conditional edge ---  run next --------------------------------------
                          ---  condition not met don't run next x
                          ---  run next --------------------------------------   ALL --- Will Run in Parallele (Interesting!) so be carefull with state updates to not mess up
                          ---  run next --------------------------------------
                          ---  condition not met don't run next x
                          ---  run next --------------------------------------

Because if this, it can be confusing with the state passed to different nodes and that is when we might really need to manage different states for different routes.
LangGraph documentation suggests to use the mapReduce design and permit us to have a specific state sending its messages to that state for example. The method used is `send` which allow us to pass the state to the next node (a specific state as here we have running parallele actions as the conditional edge condition is met for several next node to be activated):
eg:.
```python
# we just need to pass that state to the function
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]
# conditional edge
graph.add_conditional_edges("node_a", continue_to_jokes)
```

# LangGraph Checkpointers
Checkpointers are used for human interraction in the loop or if there a chat to save the messges as a `memory` so that the context is preserved.
Checkpointer are also a good way to see the state at a certain important moment if needed.
Checkpointer need thread to be passed in to keep an id that would be used to manage for example concurrent use by different user of the app. Which will maiantain different states for different users.
```python
# here we show as well that a specific thread can be called with it's ID when invoking LangChain runnables
config = {"configurable": {"thread_id": "a"}}
graph.invoke(inputs, config=config)
```
##### Checkpointer values to be passed in
values: This is the value of the state at this point in time.
next: This is a tuple of the nodes to execute next in the graph.

#### Example of a checkpointer coming from the library
```python
from langgraph.checkpoint import MemorySaver
checkpointer = MemorySaver()
# Note that we're (optionally) passing the memory when compiling the graph but better  for sophisticated wrokflows like ones including breakpoints
app = workflow.compile(checkpointer=checkpointer)
```

#### Now you can get state at a checkpoint by calling `get_state`
``` python
graph.get_state(config)
```
config: the config should contain thread_id specifying which thread to update when needed to update it.

#### use `update_state` to update the state
eg. with reducer to understand it all:
```python
from typing import TypedDict, Annotated
from operator import add
class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]

# Let's now assume the current state of the graph is
{"foo": 1, "bar": ["a"]}

# If you update the state as below:
graph.update_state(config, {"foo": 2, "bar": ["b"]})

# Then the new state of the graph will be:
{"foo": 2, "bar": ["a", "b"]}
```

The reducer keeps the previous states therefore we can track the changes while the normal state is overrided with a new value. That is why we use reducers.
After that two different behaviors:
- you call `as_node` to precise which node is considered to be one changing the state 
- OR you don't call `as_node` and then the default behavior is to consider that changes are coming from the previous node

# LangGraph Configurations

You can set the configuration at the beginning in order to be able to access to those different configs at any node point. This allow customization and use of different options like different models for example and see what is the outcome for different models and compare those like that.
```python
class ConfigSchema(TypedDict):
    llm: str

graph = StateGraph(State, config_schema=ConfigSchema)

# You can then pass this configuration into the graph using the configurable config field.
config = {"configurable": {"llm": "anthropic"}}
graph.invoke(inputs, config=config)

# You can then access and use this configuration inside a node:
def node_a(state, config):
    # here we access the config to get the llm config we want
    llm_type = config.get("configurable", {}).get("llm", "openai")
    # here we use that llm which has been saved to a variable
    llm = get_llm(llm_type)
    ...
```

# LangGraph Breakpoints

Set breakpoints either before a node executes (using interrupt_before) or after a node executes (using interrupt_after.)
This can be useful if needed to have a human interaction which would stop the graph and resume it after.
Note:
 - checkpointer (pass it at the compilation moment like memory) need to be used here in order for the graph to know the state where it should restart from
 - use `None` to restart where you stopped at
```python
# Initial run of graph
graph.invoke(inputs, config=config)

# Resume by passing in None
graph.invoke(None, config=config)
```

This is really interesting as we could stop the graph to do some other things in the application code or use other systems and have some variables updated that could later on be used by next nodes in order to have sophisticated workflow which can stop and restart while waiting for other operations to be done.

##### Example
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display

class State(TypedDict):
    input: str

def step_1(state):
    print("---Step 1---")
    pass

def step_2(state):
    print("---Step 2---")
    pass

def step_3(state):
    print("---Step 3---")
    pass

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# Set up memory
memory = MemorySaver()

# Add and put the `interrupt_before` specifiying at which step
graph = builder.compile(checkpointer=memory, interrupt_before=["step_3"])

# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

```python
# now we need a thread ID for the checkpointer before interruption at step 3
# Input
initial_input = {"input": "hello world"}
# Thread
thread = {"configurable": {"thread_id": "1"}}
# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)
# Here use is asked if we continue after the interruption at step 3
user_approval = input("Do you want to go to Step 3? (yes/no): ")
if user_approval.lower() == "yes":
    # If approved, continue the graph execution as here we use the `None` keyword as first argument, the thread ID is provided so the checkpint know where to restart from
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)else:
    print("Operation cancelled by user.")

Outputs:
{'input': 'hello world'}
---Step 1---
---Step 2---
---Step 3---
Do you want to go to Step 3? (yes/no): 
```
##### precision about `stream_mode` different values that it takes
- `"values"` which output full state
- `"updates"` which ouput only updates
- `"debug"` which output as much as possible to debug

#### `astream_events`
This is to stream event like llm calls and token tracking while those are being called inside nodes.
In the codumentation examples this is used for async functions


# LangGraph ReAct Agent

Providing multiple tools to an llm and letting the llm use those tools as much as it wants and decide on the input of those tools until it decides that it doesn't need to use any tool anymore and exists the while loop. Define in the docs as: `In this architecture, an LLM is called repeatedly in a while-loop. At each step the agent decides which tools to call, and what the inputs to those tools should be. Those tools are then executed, and the outputs are fed back into the LLM as observations. The while-loop terminates when the agent decides it is not worth calling any more tools.`

#### Example of ReAct Agent
```python
from langgraph.prebuilt import create_react_agent
```
check documentation as example is too long: [Example of ReAct Agent Docs](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent)
Short example:
- create a ReAct Type of Agent Tool
```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import MessageGraph
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def divide(a: float, b: float) -> int:
    """Return a / b."""
    return a / b

llm = ChatAnthropic(model="claude-3-haiku-20240307")
tools = [divide]
graph_builder = MessageGraph()
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("chatbot", llm.bind_tools(tools))
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges(
...     "chatbot", tools_condition
... )
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile()
graph.invoke([("user", "What's 329993 divided by 13662?")])
```
- Then create a validation node
```python
from typing import Literal
from langchain_anthropic import ChatAnthropic
from langchain_core.pydantic_v1 import BaseModel, validator
from langgraph.graph import END, START, MessageGraph
from langgraph.prebuilt import ValidationNode

class SelectNumber(BaseModel):
    a: int

    @validator("a")
    def a_must_be_meaningful(cls, v):
        if v != 37:
            raise ValueError("Only 37 is allowed")
        return v

builder = MessageGraph()
llm = ChatAnthropic(model="claude-3-haiku-20240307").bind_tools([SelectNumber])
builder.add_node("model", llm)
builder.add_node("validation", ValidationNode([SelectNumber]))
builder.add_edge(START, "model")

def should_validate(state: list) -> Literal["validation", "__end__"]:
    if state[-1].tool_calls:
        return "validation"
    return END

builder.add_conditional_edges("model", should_validate)

def should_reprompt(state: list) -> Literal["model", "__end__"]:
    for msg in state[::-1]:
        # None of the tool calls were errors
        if msg.type == "ai":
            return END
        if msg.additional_kwargs.get("is_error"):
            return "model"
    return END

builder.add_conditional_edges("validation", should_reprompt)
graph = builder.compile()
res = graph.invoke(("user", "Select a number, any number"))
# Show the retry logic
for msg in res:
   msg.pretty_print()

Outputs:
================================ Human Message =================================
Select a number, any number
================================== Ai Message ==================================
[{'id': 'toolu_01JSjT9Pq8hGmTgmMPc6KnvM', 'input': {'a': 42}, 'name': 'SelectNumber', 'type': 'tool_use'}]
Tool Calls:
SelectNumber (toolu_01JSjT9Pq8hGmTgmMPc6KnvM)
Call ID: toolu_01JSjT9Pq8hGmTgmMPc6KnvM
Args:
    a: 42
================================= Tool Message =================================
Name: SelectNumber
ValidationError(model='SelectNumber', errors=[{'loc': ('a',), 'msg': 'Only 37 is allowed', 'type': 'value_error'}])
Respond after fixing all validation errors.
================================== Ai Message ==================================
[{'id': 'toolu_01PkxSVxNxc5wqwCPW1FiSmV', 'input': {'a': 37}, 'name': 'SelectNumber', 'type': 'tool_use'}]
Tool Calls:
SelectNumber (toolu_01PkxSVxNxc5wqwCPW1FiSmV)
Call ID: toolu_01PkxSVxNxc5wqwCPW1FiSmV
Args:
    a: 37
================================= Tool Message =================================
Name: SelectNumber
{"a": 37}
```

#### workaround found by a user if with_structured_output() can't be used
As here we can't use `with_structured_output()` to create our tools we can use `parser_chain` using the `PydanticOutputParser` to get the structured output:
```python
graph = create_react_agent(model=llm,tools=self.tools,messages_modifier=prompt) | RunnableLambda(lambda data: {"text": str(data["messages"][-1].content)}) | parser_chain
```

# What if we want the LLM to run the script that it has created and not arm the server running the app?

##### Running in a Separate Process with Limited Permissions
Subprocess Module: Running the code in a subprocess with restricted permissions can isolate the execution from the main process.

```python
import subprocess

try:
    result = subprocess.run(['python3', 'script.py'], capture_output=True, text=True, timeout=10)
    print(result.stdout)
except subprocess.TimeoutExpired:
    print("The script took too long to complete.")
```

##### Using Docker for Sandbox (my preferred one)
```bash
# nano Dockerfile
# Dockerfile
FROM python:3.9-slim
# Copy the requirements.txt file into the container
COPY requirements.txt .
# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt
# script that we want to run inside docker
COPY app.py /app/app.py
# copy .dynamic.env file special for agent if needed (optional)
# COPY .dynamic.env /app/.dynamic.env
WORKDIR /app
CMD ["python3", "app.py"]
# Build and run (this step will be replaced by a script capturing output)
docker build -t sandbox-python .
docker run --rm sandbox-python



# BUT, we will use the following script so that the Agent can capture the output of the script ran in docker using `subprocess`
import subprocess

def run_script_in_docker(dockerfile_name_to_run_script: str, agent_script_file_name: str) -> Tuple[str, str]:

    # Write the script content to a file
    with open("agent_created_script_to_execute_in_docker.py", "w", encoding="utf-8") as going_to_docker_script_file, open(agent_script_file_name, "r", encoding="utf-8") as script_content:
        going_to_docker_script_file.write(script_content.read())

    # Create the dockerfile for the agent to run the script inside docker
    with open(dockerfile_name_to_run_script, "w", encoding="utf-8") as docker_file:
        docker_file.write("FROM python:3.9-slim\n")
        docker_file.write("COPY requirements.txt .\n")
        docker_file.write("RUN pip install --no-cache-dir -r requirements.txt\n")
        docker_file.write("COPY agent_created_script_to_execute_in_docker.py /app/agent_created_script_to_execute_in_docker.py\n")
        docker_file.write("COPY .dynamic.env /app/.dynamic.env\n")
        docker_file.write("WORKDIR /app\n")
        docker_file.write('CMD ["python", "agent_created_script_to_execute_in_docker.py"]')

    try:
        # Build the Docker image
        build_command = ['docker', 'build', '-t', 'sandbox-python', '-f', f'{dockerfile_name_to_run_script}', '.']
        subprocess.run(build_command, check=True)

        # Run the Docker container and capture the output
        run_command = ['docker', 'run', '--rm', 'sandbox-python']
        result = subprocess.run(run_command, capture_output=True, text=True)

        stdout, stderr = result.stdout, result.stderr
        with open("agent_docker_script_execution_result.md", "w", encoding="utf-8") as script_execution_result:
          script_execution_result.write("# Python script executed in docker, this is the result of captured stdout and stderr")
          script_execution_result.write("""
            This is the result after the execution of the code
            Returns:
            stdout str: the standard output of the script execution which runs in docker, therefore, we capture the stdout to know if the script output is as expected. You need to analyze it and see why it is empty if emppty, why it is not as expected to suggest code fix, and if the script executes correctly, get this stdout value and answer using markdown ```python ``` saying just one word: OK'
            stderr str: the standard error of the script execution. If this value is not empty answer witht the content of the value with a suggestion in how to fix it. Answer using mardown and put a JSON of the error message with key ERROR between this ```python ```. 
          """)
          script_execution_result.write(f"\n\nstdout: {stdout}\nstderr: {stderr}")

    except subprocess.CalledProcessError as e:
        stdout, stderr = '', str(e)
    finally:
        # Remove the Docker image
        cleanup_command = ['docker', 'rmi', '-f', 'sandbox-python']
        subprocess.run(cleanup_command, check=False)

    # Return the captured output
    return stdout, stderr

# Example script content created by the LLM
script_content = """
print('Hello, World!')
"""

stdout, stderr = run_script_in_docker(script_content)

print("Output:")
print(stdout)
print("Errors:")
print(stderr)


# PGVector and Postgreql linking

#### How to link PGVector collection to the corresponding data from Postrgesql database table.
##### Solution is to use a UUID which will be also embedded:
- The Posgreql database have a column with unique identifier UUID
- The column containing the cleaned data that we want to embed will have a UUID
- Embed the data with the UUID
- Retrieve on user query
- Then parse the UUID to fetch the database row and get the column that we want


# Matching Query Semantic vs Exact
When a user sends its query we need a way to check if we can just use the cache to answer or if we need to perform a vector search.
For that if we decide to check on the cache before making any vector search, we need to find a similar query:
- Exact match: we find the smae query and fetch the content stored
- Semantic mecth: we use embedding of the query to search for queries int he cache with are similar (I prefer this one)
 
But we will implement a function that does both, if it founds same query, it fetches from cache but will also check if there is semantic similiraty relevance. If not,the last chance will be to go to the vector db search.

When all fails, even the vector db search, LLM can use other tools to search online.
 
____________________________________________________

# retrieve_relevant_vectors object returned:
List[Dict[str,any]]: [{'UUID': info['UUID'], 'content': info['content'], 'score': score}]

# answer_retriever: Object embedded fields plus db row corresponding to that object. So here we could use the 'row_data' field to get other info needed for answer like `title` and `doc_name`
List[Dict[str,any]]: [{ 'UUID': doc['UUID'], 'score': doc['score'], 'content': doc['content'], 'row_data': document},]
 
# similarity_search_with_score
returns a list ordered from the highest score to the lowest score.
The higher the score, the more relevant is the answer.
 
# LangGraph custom state
You just need to create your custom states and use those in the different tools or nodes functions.
Therefore, you can create, update, delete those vars as your app is processing different actions
```python
from typing import TypedDict

class CustomState(TypedDict):
    input: str
    steps_completed: int
    data: Dict[str, str]


def create_initial_state(input_value: str) -> CustomState:
    return CustomState(input=input_value, steps_completed=0, data={})

# example of function that uses the state in it:
def step_1(state: CustomState):
    (...)
    Some logic
    (...)
    Some logic
    (...)
    print("---Step 1---")
    state['steps_completed'] += 1
    state['data']['step_1'] = "Processed Step 1"
    return state

(...)
Node creation
(...)
Tools and conditional edges etc...
(...)
compilation of graph
graph=...

from langgraph import GraphRunner

# Create the initial state
initial_state = create_initial_state("Initial Input")

# Create the graph runner to run the langgraph compile object. Here `graph` is the compiled langgraph graph
runner = GraphRunner(graph)

# Run the graph
final_state = runner.run(initial_state)

print("Final State:", final_state)
```

# Langchain prompts
- PromptTemplate.from_template(template=a_template_str_with_LCEL_{variable}) = use this with LCEL {variable} and when you invoke pass in the variable, eg.: ...invoke({"variable": "value"})
- PromptTemplate(template=a_template_str_with_LCEL_{variable}, input_variables=["variable",]) Or use here `.format({"variable": "value"})` or when using invoke pass in the variable. eg.: ...invoke({"variable": "value"}) 

- advanced prompts
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
) 

 # use of invoke, run and predict
- Use invoke() when you have a full chain involving multiple components (prompt template, LLM, parser) and need to execute the entire chain. so here for chains for example
- Use run() when you are dealing with a StateGraph or similar workflow structure and need to execute the defined sequence of operations. so here for full graph run for example
- Use predict() when you need to generate a prediction from a model based on specific input. so here just for input/output from model
 
 
 # ISSUES
 
## PromptTemplate object
One import used did not return a `PromptTemplate` but a `str` which is annoying
- from langchain_core.prompts.prompt import PromptTemplate : type = `str`
- from langchain.prompts import PromptTemplate : type = `PromptTemplate`
 
## Incorrect Type Checking:
- The original type checks were checking if the formatted prompt was of type str instead of PromptTemplate.
```python
if not isinstance(formatted_system_prompt, str):
    raise ValueError("formatted_system_prompt is not an instance of PromptTemplate")
if not isinstance(formatted_human_prompt, str):
    raise ValueError("formatted_human_prompt is not an instance of PromptTemplate")
```
This check should verify if the prompt is an instance of PromptTemplate, not str.

- Correct Type Checking:
The corrected code properly checks if the formatted prompts are instances of PromptTemplate.
```python
if not isinstance(formatted_system_prompt, PromptTemplate):
    raise ValueError("formatted_system_prompt is not an instance of PromptTemplate")
if not isinstance(formatted_human_prompt, PromptTemplate):
    raise ValueError("formatted_human_prompt is not an instance of PromptTemplate")
```

## Returning String Instead of PromptTemplate:

- The prompt_creation function returned a str when formatting the template, but it should always return a PromptTemplate.
```python
if input_variables:
    return PromptTemplate(
        template=target_prompt["template"],
        input_variables=input_variables
    ).format(**kwargs)
else:
    return PromptTemplate(
        template=target_prompt["template"],
        input_variables=input_variables
    )
```
This code incorrectly handles the formatting and type of the returned object.

- Returning PromptTemplate Instead of str:
The prompt_creation function ensures that it always returns a PromptTemplate, even after formatting.
```python
def prompt_creation(target_prompt: Dict[str, Any], **kwargs: Any) -> PromptTemplate:
    input_variables = target_prompt.get("input_variables", [])
    
    prompt = PromptTemplate(
        template=target_prompt["template"],
        input_variables=input_variables
    )
    
    formatted_template = prompt.format(**kwargs) if input_variables else target_prompt["template"]
    
    return PromptTemplate(
        template=formatted_template,
        input_variables=[]
    )
```

## Improper Handling of Empty Input Variables:

- The original code does not correctly handle cases when input_variables is empty, leading to potential errors during formatting or type mismatches.

- Proper Handling of Empty Input Variables:
The corrected code handles cases where input_variables might be empty by defaulting to an empty list.
```python
input_variables = target_prompt.get("input_variables", [])
```

## Incorrect Prompt Initialization:

- When initializing SystemMessagePromptTemplate and HumanMessagePromptTemplate, the prompt should be a PromptTemplate, not a dictionary or string.
```python
system_message_prompt = SystemMessagePromptTemplate(prompt={"template": formatted_system_prompt.template})
```

## Correct Prompt Initialization:

- The corrected code initializes SystemMessagePromptTemplate and HumanMessagePromptTemplate with the PromptTemplate directly.
```python
system_message_prompt = SystemMessagePromptTemplate(prompt=formatted_system_prompt)
human_message_prompt = HumanMessagePromptTemplate(prompt=formatted_human_prompt)
```

## Corrected full functions and imports"
```python
from typing import Dict, Any
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate

def prompt_creation(target_prompt: Dict[str, Any], **kwargs: Any) -> PromptTemplate:
    input_variables = target_prompt.get("input_variables", [])
    
    prompt = PromptTemplate(
        template=target_prompt["template"],
        input_variables=input_variables
    )
    
    formatted_template = prompt.format(**kwargs) if input_variables else target_prompt["template"]
    
    return PromptTemplate(
        template=formatted_template,
        input_variables=[]
    )

def chat_prompt_creation(system_prompt: Dict[str, Any], human_prompt: Dict[str, Any], *args: Dict[str, Any], **kwargs: Any) -> ChatPromptTemplate:
    ai_message_prompt_list = []
  
    # Define system prompt and user prompt
    formatted_system_prompt = prompt_creation(system_prompt, **kwargs)
    formatted_human_prompt = prompt_creation(human_prompt, **kwargs)

    # Ensure the formatted prompts are instances of PromptTemplate
    if not isinstance(formatted_system_prompt, PromptTemplate):
        raise ValueError("formatted_system_prompt is not an instance of PromptTemplate")
    if not isinstance(formatted_human_prompt, PromptTemplate):
        raise ValueError("formatted_human_prompt is not an instance of PromptTemplate")

    system_message_prompt = SystemMessagePromptTemplate(prompt=formatted_system_prompt)
    human_message_prompt = HumanMessagePromptTemplate(prompt=formatted_human_prompt)
    print("system message prompt: ", system_message_prompt)
    print("human message prompt: ", human_message_prompt)
  
    # Can be used as an example of an answer saying "This is an example of answer..."
    for arg in args:
        formatted_ai_message_prompt = prompt_creation(arg, **kwargs)
        if not isinstance(formatted_ai_message_prompt, PromptTemplate):
            raise ValueError("formatted_ai_message_prompt is not an instance of PromptTemplate")
        ai_message_prompt = AIMessagePromptTemplate(prompt=formatted_ai_message_prompt)
        ai_message_prompt_list.append(ai_message_prompt)
  
    # Instantiate the prompt for the chat
    if ai_message_prompt_list:
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt, *ai_message_prompt_list])
    else:
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
  
    return chat_prompt
```



### Lot of confusion around prompt and errors: The following is from the doc, just follow that!!!

##### NORMAL VERSION
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# get the template part of the prompt
prompt = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + "\n\nand in {language}"
)
# get input variable if any for the prompt template part if using .format
prompt.format(topic="sports", language="spanish")

# for example for the model
model = ChatGroq(...)

# or if not using .format just put prompt in LLMChain and put input vars in run()
chain = LLMChain(llm=model, prompt=prompt)
chain.run(topic="sports", language="spanish")

# make the function for that: prompt_template is going to be imported from the prompt module
def make_normal_prompt(prompt_template: Dict, **kwargs):
  model = ChatGroq(...)
  prompt = (
    PromptTemplate.from_template(prompt_template["template"])
  )
  
  if prompt_template["input_variables"]:
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.run(**kwargs)
    return response.content.split("```")[1].strip("python").strip()

  chain = LLMChain(llm=model, prompt=prompt)
  response = chain.run()
  return response.content.split("```")[1].strip("python").strip()

_____________________________

##### CHAT VERSION
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


prompt = SystemMessage(content="You are a nice pirate")
new_prompt = (
    prompt + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"
)

new_prompt.format_messages(input="i said hi")

Should Output an Object Like:
`[SystemMessage(content='You are a nice pirate', additional_kwargs={}),
 HumanMessage(content='hi', additional_kwargs={}, example=False),
 AIMessage(content='what?', additional_kwargs={}, example=False),
 HumanMessage(content='i said hi', additional_kwargs={}, example=False)]`
 
 chain = LLMChain(llm=model, prompt=new_prompt)

chain.run("i said hi")

### Errors to handle list:
#### GROQ Errors
- calling api: groq.InternalServerError: Error code: 502 - {'error': {'type': 'internal_server_error', 'code': 'service_unavailable'}}   -> solution implement retries exponential and with max retry times and prepare fallback local lmstudio or ollama maybe... and trace to know when this happen

#### LLM calls Chaining vs direct call
Direct call is better as no issues with formatting of input, LLM understands more what to do.
- for the query:
```bash
           "system",
             """
              - Identify if the user query have a .pdf file in it, or an url, or just text, also if there is any clear question to rephrase it in a easy way for an llm to be able to retrieve it easily in a vector db. 
              - If non of those found or any of those just put an empty string as value in the corresponding key in the response schema.
              - User might not clearly write the url if any. It could be identified when having this patterm: `<docmain_name>.<TLD: top level domain>`. Make sure to analyze query thouroughly to not miss any url.
              - Put your answer in this schema:
                {
                  'url': <url in user query make sure that it always has https. and get url from the query even if user omits the http>,
                  'pdf': <pdf full name or path in user query>,
                  'text': <if text only without any webpage url or pdf file .pdf path or file just put here the user query>,
                  'question': <if quesiton identified in user query rephrase it in an easy way to retrieve data from vector database search without the filename and in a good english grammatical way to ask question>
                }
              Answer only with the schema in markdown between ```python ```.
             """,
             "human": "I have been in the ecommerce site chikarahouses.com and want to know if they have nice product for my Japanese garden"
```
- chaining and then invoke returns:
```bash
{
    'url': 'https' + user_query.split('://')[-1] if any(x in user_query for x in ['http', 'www']) else '',
    'pdf': user_query.split('.pdf')[-2] if '.pdf' in user_query else '',
    'text': user_query if not ('http' in user_query or '.pdf' in user_query) else user_query.split('.pdf')[0] if '.pdf' in user_query else user_query.split('/')[-1],
    'question': re.sub(r'[\[\]\(\)\{\}\|@\\/:;><\']+', ' ', user_query.strip()).strip() if any(x in user_query for x in ['how', 'what', 'where', 'when', 'who', 'which', 'why', 'is', 'are', 'does', 'do', 'did', 'was', 'were', 'has', 'have', 'had']) else ''
}
```
- Direct call retuns:
```bash
{
  'url': 'https://chikarahouses.com',
  'pdf': '',
  'text': 'I have been in the ecommerce site chikarahouses com and want to know if they have nice product for my Japanese garden',
  'question': 'Do they have nice products for a Japanese garden at Chikara Houses?'
}
```

##### Structure of document data when pdf content is parsed
- function : `+pdf_to_sections`
```code

 {
  'text': 'Exploring the cellular and molecular mechanisms of temperature sensation. One of my favourite things to learn in high school chemistry was how temperature workedâ \\200\\224the average kinetic energy of substances being responsible for how hot or cold we m ight feel something to be. This was simple enough. Fast molecules that bump a lot = hotter. Slow molecules that donâ\\200\\231t bump as often = colder. But that doesnâ\\200\\231t exactly explain how our brain can sense and relay information about the temperature of a substance . When you pick up an ice cube, how does your brain know what you’re touching is cold? How do we turn that measure of kinetic energy into signals that we can regularly identify as co', 
  'section': 'Feb 22, 2024', 
  'document': './doc/feel_temperature.pdf'
  }, 
  {
   'text': 'Thatâ\\200\\231s what Iâ\\200\\231m aiming to explain today, and having taken a course in cellu lar and molecular biology, this has only just started to make sense to me.', 
   'section': 'ld?', 
   'document': './doc/feel_temperature.pdf'
   }, 
  {
   'text': 'The main challenge here is figuring out how thermal and kinetic energy is converted into el ectrical signals that can inform the brain of the temperature of a substance. When it comes to receiving or transmitting any sort of signals, it’s a safe assumption that receptors of some sort play a massive role in the travel of information. Thatâ\\200\\231s no different he', 
   'section': 'Thermoreceptors', 
   'document': './doc/feel_temperature.pdf'
   }, 
   {
    'text': 'A simplified diagram of thermoreceptors in the dermis (layer of skin) A simplified diagram of thermoreceptors in the dermis (layer of skin) | Source Thermoreceptors are free nerve endings (FNE), which extend until the mid-epidermis, or the outermost layer of our skin. The ganglionsâ\\200\\224the structures responsible for receiving information from extracellular stimuliâ\\200\\224are not enclosed within a membrane, which a llows them to detect various physical stimuli through interactions involving our skin. Additionally, we have 2 types of thermoreceptors, cold and warm receptors, that can be vari ed in concentration throughout the body. Iâ\\200\\231m sure youâ\\200\\231ve noticed that your ears and face get excessively cold in the winter, thatâ\\200\\231s due to a greater presence', 
     'section': 'Introducing, the thermoreceptor!', 
     'document': './doc/feel_temperature.pdf'
    }, 
   {
    'text': 'To understand how these FNE thermoreceptors work, it is important to first understand a lit tle bit about neurotransmission. The strength of a signal processed by the brain is depende nt on the frequency of a neuronâ\\200\\231s firing. Of course, we are always interacting with substances and objects, which all have surface temperatures that our thermoreceptors detec t. However, these tend to be at room temperature, corresponding to normal firing rates. Int eractions with cold or hot stimuli change the firing rate of their corresponding thermorece', 
     'section': 'of cold receptors in those areas.', 
     'document': './doc/feel_temperature.pdf'
    }, 
    {
     'text': 'Touching something with a temperature between around 5â\\200\\22330Â°C will increase cold rec eptor firing and decrease warm receptor firing. Accordingly, stimuli of temperature between 30â\\200\\22345Â°C will increase warm receptor firing and decrease cold receptor firing. Thi s changing pattern of neural impulses by FNEs is what indicates to your brain (and therefor e, you) that youâ\\200\\231re touching something cold or hot. In fact, when you touch something hot enough to hurt, another type of FNEs called nocicepto rs are activated, which signals to your brain that something youâ\\200\\231re touching is cau sing you pain. That, coupled with a high warm receptor firing rate, makes you aware that yo u might burn yourself if you keep in contact with the given substance.feel_temperature.txt Thu May 09 19:02:26 2024 2', 
      'section': 'ptors.', 
      'document': './doc/feel_temperature.pdf'
     }, 
     {
      'text': 'So, weâ\\200\\231ve covered how we feel one type of hot, but what about that burning feeling', 
      'section': 'Ion Channels', 
      'document': './doc/feel_temperature.pdf'
     }, 
     {
      'text': 'A diagram of TRP ion channels sensitive to temperature and chemical heat sources A diagram of TRP ion channels sensitive to temperature and chemical heat sources | Source The heat of a chilli pepper is slightly different from what weâ\\200\\231ve talked about befo re. These plants contain capsaicin, a chemical agent that acts on interior surfaces in your mouth (and tongue, especially) to make it feel like your tongue is burning. As aforementio ned, for this change in sensation in your mouth to happen, there must be some receptor invo lved that can turn a physical or chemical stimulus into signals your brain can decipher. In this case, this receptor happens to be the TRPV1 receptor (crazy name, just think of it as a warm receptor), which causes an influx of sodium and calcium into nerve cells. This init iates the firing of associated nerves that indicate to the brain that something is on fire in your mouth, figuratively but Iâ\\200\\231m sure that’s what it feels like, more or less. T RPM8 receptors are similar in mechanism, but instead conduct messages about cold stimuli wh en cooling agents like methanol bind to the receptors. Heat and temperatures are so fascinating to me, how we can perceive so strongly the change in movements of such small molecules. If you want to learn more about different types of re ceptors that signal for responses to other physical stimuli like pressure or touch, check o ut the following short video that inspired me to make this article!', 
       'section': 'right after eating an unexpectedly spicy chilli?', 
       'document': './doc/feel_temperature.pdf'
      }
  ]
```

#### Next
- change all ```python ``` by ```markdown ``` in prompts as the stupid llm sometimes interpret the ```python ```` as being creation of python code...
- see if padf_parser shouldn't be simplified and just use same process as webpage parser and get all text and then chunk it and then summarize it... not sure yet as pdf parser that we have take into consideration tables and more ... we keep it like that for the moment
- test this function: process_query form app.py and make all necessary imports to app.py
- export prompts to the prompt file and import those to be used in the functions that summarize text and make tiles 
