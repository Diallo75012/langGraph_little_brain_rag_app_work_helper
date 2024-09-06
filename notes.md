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
# Note that we're (optionally) passing the memory when compiling the graph but better for sophisticated workflows like ones including breakpoints
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
In the documentation examples this is used for async functions


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

# beware of `langchain_community.vectorstores.pgvector` metadata storage as JSONB instead of JSON
```bash
# warning message
/home/creditizens/langgraph/langgraph_venv/lib/python3.10/site-packages/langchain_community/vectorstores/pgvector.py:328: LangChainPendingDeprecationWarning: Please use JSONB instead of JSON for metadata. This change will allow for more efficient querying that involves filtering based on metadata.Please note that filtering operators have been changed when using JSOB metadata to be prefixed with a $ sign to avoid name collisions with columns. If you're using an existing database, you will need to create adb migration for your metadata column to be JSONB and update your queries to use the new operators. 
  warn_deprecated(
```

# Store and query embeddings metadata: `full example`
Interesting when wanting to filter the content and narrow down the search to specific parts of the documents and to organize those like that.
It could be a table column having those metadata and the collection having the embedded target document with same metadata. So two different agents or nodes could be used, one that search the metada using user query and the other taking that result to perform a vector search. The last would send it to another node for analysis and answer formating or next step actions.

To store vectors as JSONB in a PostgreSQL table, especially when dealing with high-dimensional vectors like 4096 dimensions, you would create a table with a JSONB column and an additional column to store the vectors. Here's how you can do this using the xample of a table storage for everything to make it easily understandable:


```sql
# table creation with JSONB fields and 4096(ollama mistral:7b dimensions) vector dimension field
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    doc_name TEXT,
    title TEXT,
    content JSONB,
    embedding VECTOR(4096),
    metadata JSONB
);

# insert data using `::JSONB`
INSERT INTO documents (id, doc_name, title, content, embedding, metadata)
VALUES (
    gen_random_uuid(),
    'example_doc.pdf',
    'An Example Document',
    '{"paragraph1": "This is an example paragraph.", "paragraph2": "This is another example."}',
    '[0.1, 0.2, 0.3, ..., 0.4096]'::VECTOR(4096),
    '{"author": "John Doe", "tags": ["example", "demo"]}'::JSONB
);

# query metada using `@>`
SELECT * 
FROM documents 
WHERE metadata @> '{"author": "John Doe"}';

# perform vector similarity using `<=>` and `::VECTOR(4096)`
SELECT id, title, embedding
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, 0.3, ..., 0.4096]'::VECTOR(4096)
LIMIT 10;
```
- The <=> operator is used for vector similarity search.
- The JSONB @> operator is used to check if the JSONB column contains the specified key-value pair.

**For the moment**
- Can search using metadata `key` using: db.similarity_search_with_score(query, filter={"metadata.key": "value"}) for the moment as we are using PGVector collection which is different way to store embeddings and not a Postgresql normal table column but a separate entity where embeddings are stored
**BUT if using above example**
- if using postgresql table column insstead of collection(pgvector) to store metadata as JSONB: use `$` in front of metadata key `db.similarity_search_with_score(query, filter={"metadata.$key": "value"})`

**Note about table stored vector**
- PostgreSQL does not generate embeddings: You must use an embedding model to create the vector representation of your data.
- PostgreSQL VECTOR(4096) column: This is simply a storage type for your pre-computed vectors; it does not perform any transformations on the data.
- Use here ollama mistral:7b to change data into a vector before storing it in the db for example.


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
 
## Database storage `''` vs  `""`

- Here the issue is that the values are stored to db using double quotes `""` while those should be encapsulated in single quote `''`
```bash
Error inserting row 0: column "https://chikarahouses.com" does not exist
LINE 3:                 WHERE doc_name = "https://chikarahouses.com"...
                                         ^
```

- Here we get single quotes `''` for valeu stored in db and for `table_name` we can use sql.Identifier which will use `""`
```sql
cursor.execute(sql.SQL("""
        INSERT INTO {} (id, doc_name, title, content, retrieved)
        VALUES (%s, %s, %s, %s, %s)
    """).format(sql.Identifier(table_name)), 
    [row["id"], row["doc_name"], row["title"], row["content"], row["retrieved"]])
    conn.commit()
```

## Data serialization saved to state which has limit of storage
We can have large documents and we want to store those in db by creating a dataframe beforehands.
This dataframe stored to state to pass it to the next node which is going to store it to the database.
- But the dtaframe can be too large and we used JSON serialization but it is not optimal for size.
- We are using for the moment parquet stored to memory which takes less space (pyarrow)
- But instead of having different nodes to do those jobs we could just make that node do those two tasks parsing document and storing to db to prevent large state variables

```python
import pyarrow as pa
import pyarrow.parquet as pq

# Serialize the dataframe to a Parquet format stored in memory
def serialize_dataframe(df):
    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()

# Deserialize the dataframe from the Parquet format
def deserialize_dataframe(data):
    buffer = pa.BufferReader(data)
    table = pq.read_table(buffer)
    return table.to_pandas()
```

Attempting to serialize binary or non-UTF-8 encoded data into a field that expects text content, leading to encoding errors
therefore, we need to find another solution to make it easy. I though using a file to store the dataframe and that is what we are going to do.
- store df to file
- store path location of that file in the lanngraph states message of the node.
- this way we can have separate nodes for separate tasks

```python
import uuid

def save_dataframe(df, directory="dataframes"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(directory, f"{file_id}.parquet")
    df.to_parquet(file_path)
    return file_path

def load_dataframe(file_path):
    return pd.read_parquet(file_path)
```

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

## Final prompt structure will be:
- simplification of prompt structure and uniformization
```code
# to simplify all prompts will have this structure and we will pull or fill what is needed for easy llm call
<function_name>_prompt = {
  "system": {
    "template": "", 
    "input_variables": {}
  },
  "human": {
    "template": "", 
    "input_variables": {}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}
```

# use of invoke using the prompt
```python
# where prompt_text is targetting what is needed eg: `prompt_text = <function_name>_prompt["system"]["template"]`
response = llm.invoke(prompt_text)
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

## Groq internal server error
We need to handle this error if it happens
```python
groq.InternalServerError: Error code: 503 - {'error': {'message': 'Service Unavailable', 'type': 'internal_server_error'}}
```
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
- change all ```python ``` by ```markdown ``` in prompts as the stupid llm sometimes interpret the ```python ```` as being creation of python code... - OK
- see if pdf_parser shouldn't be simplified and just use same process as webpage parser and get all text and then chunk it and then summarize it... not sure yet as pdf parser that we have take into consideration tables and more ... we keep it like that for the moment  
- test this function: process_query form app.py and make all necessary imports to app.py - OK
- Where is the function that stores to the DB the url/pdf fetched - OK function created need to be tested
- export prompts to the prompt file and import those to be used in the functions that summarize text and make tiles - OK


# PSQL

### postgresql 
```bash
# connect
sudo -u postgres psql
```

```sql
# create user
CREATE USER creditizens WITH PASSWORD 'your_secure_password';
# set priviledges for user to have hands on tables and pgvector
ALTER USER creditizens CREATEDB;
GRANT ALL PRIVILEGES ON DATABASE your_database_name TO creditizens;
GRANT USAGE ON SCHEMA public TO creditizens;
GRANT CREATE ON SCHEMA public TO creditizens;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO creditizens;
# ensure user can activate pgvector
GRANT USAGE ON SCHEMA vector TO creditizens;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA vector TO creditizens;
```

```sql
# list all users
\du
# list all databases
\l
```

```sql
# check priviledge of specific user on specific database
SELECT grantee, privilege_type 
FROM information_schema.role_table_grants 
WHERE table_catalog = 'your_database_name' 
AND grantee = 'your_username';
```

```sql
# list all databases and their owners
SELECT datname AS "Database",
       pg_catalog.pg_get_userbyid(datdba) AS "Owner"
FROM   pg_catalog.pg_database
ORDER BY 1;
```

```sql
# get detailed information about user privileges on all databases
SELECT datname AS "Database",
       pg_catalog.pg_get_userbyid(datdba) AS "Owner",
       pg_catalog.array_to_string(datacl, E'\n') AS "Access Privileges"
FROM   pg_catalog.pg_database
ORDER BY 1;
```

```bash
# terminal command for scripting eg.
# eg.: 2
psql -U postgres -c "\du"
# eg.: 1
psql -U postgres -d your_database_name -c "YOUR_SQL_QUERY_HERE"
```

```bash
# check collection creation:
psql -U creditizens -d creditizens_vector_db -c "SELECT relname FROM pg_class WHERE relname = 'langgraph_collection_test';"
```

```bash
# describe table structure:
psql -U creditizens -d creditizens_vector_db -c "\d+ langgraph_collection_test"
```

```bash
# check collection content:
psql -U creditizens -d creditizens_vector_db -c "SELECT * FROM langgraph_collection_test LIMIT 10;"
```

```bash
# check if collection has been created:
psql -U creditizens -d creditizens_vector_db -c "SELECT relname FROM pg_class WHERE relname = 'langgraph_collection_test';"

```

```bash
# delete all content from some tables, can add `CASCADE` at the end if thsoe tabled have foreign keys
TRUNCATE TABLE documents, langchain_pg_collection, langchain_pg_embedding;
```

```bash
# list users
\du
# list databases
\l
# select database
\c creditizens_vector_db;
# list for tables in that db
\dt
```

#### psycopg2 (get also psycopg2-binary) security against SQL injection

```python
import psycopg2
from psycopg2 import sql
# then: Using sql.SQL and related classes is recommended when you need to insert identifiers like table names or column names dynamically. This ensures that these values are safely quoted and prevents SQL injection
```

# Next
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one.
- test function that stores data in database- OK
- create all functions that each nodes will need (tools and other functions)
- create states and use workflow functions to save in each state, name the state the name of the node and the name of the edge fo easy logic and limit confusion, therefore, separate states (have many mini states) and find a way to empty those at the right moment with one function at the end of graph or some depending on the logic.
- See if you need function to get rid of DB info. Create the function that resets the db to zero and the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB.

## `os.system("<terminal command>")` vs subprocess
- the `os.system("<terminal command>")`: only returns the exit status of the command, not the output of the command itself.

- subprocess is going to be able to get the stdout and stderr
eg:
```python
import subprocess

result = subprocess.run(
    ["psql", "-U", "creditizens", "-d", "creditizens_vector_db", "-c", "SELECT * FROM test_doc_recorded"],
    capture_output=True,
    text=True
)

print(result.stdout)
```

# RETURNED OUPUT FOR CERTAIN FUNCTIONS
- eg.: Chunk size = 4000 `create_chunks_from_db_data(fetch_data_from_database, 4000)` 
code```
[
  [
    {'UUID': '0d3531c5-bf35-4e3b-b695-b2c2eaf9f22b', 'content': "To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on neuron firing frequency. We interact with objects at room temperature, which causes normal firing rates in thermoreceptors. However, cold or hot stimuli change the firing rate of their corresponding thermoreceptors."}, 
    {'UUID': '112074b3-09e0-4759-b32e-c1df851fa219', 'content': "To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on neuron firing frequency. We interact with objects at room temperature, which causes normal firing rates in thermoreceptors. However, cold or hot stimuli alter the firing rate of their corresponding thermoreceptors."}, 
    {'UUID': '1e6a1648-227f-4843-bf2b-7073a192f8cf', 'content': 'Thermoreceptors in the dermis are free nerve endings that detect temperature changes. They extend to the mid-epidermis and interact with extracellular stimuli. There are two types: cold and warm receptors, with varying concentrations throughout the body. A higher concentration in ears and face can cause excessive coldness in winter.'}, 
    {'UUID': '240c0f7e-1d39-423c-b03d-91dc4aee8473', 'content': 'The challenge is converting thermal and kinetic energy into electrical signals for the brain. Receptors are key in signal transmission, and they play a massive role here as well.'}, 
    {'UUID': '2fdd1c07-c465-4b5e-887b-d7856d179325', 'content': "Exploring the cellular and molecular mechanisms of temperature sensation goes beyond the average kinetic energy of substances. When you touch something cold, like an ice cube, how does your brain know it's cold? This involves understanding how the measure of kinetic energy is converted into signals that our brain can identify as cold.\n\nAt the molecular level, temperature sensation is mediated by ion channels in specialized nerve endings called nociceptors. These channels are sensitive to changes in temperature and open or close in response to these changes, allowing ions to flow in and out of the cell. This ion flow generates electrical signals that are transmitted to the brain, where they are interpreted as temperature sensations.\n\nOne example of a temperature-sensitive ion channel is TRPM8, which is activated by cold temperatures. When the temperature drops, TRPM8 channels open, allowing calcium ions to flow into the cell. This ion flow triggers a series of events that ultimately leads to the generation of an electrical signal that is transmitted to the brain.\n\nOverall, the cellular and molecular mechanisms of temperature sensation involve the detection of changes in kinetic energy by specialized ion channels in nerve endings, which convert these changes into electrical signals that are transmitted to the brain."},
    {'UUID': '569ce0d4-d21e-4f7f-a8a3-ffc8c30a4e1e', 'content': 'The challenge is converting thermal and kinetic energy into electrical signals for the brain. Receptors are key in signal transmission, and they play a massive role here as well.'}, 
    {'UUID': '583686a4-3708-4333-8507-1fe84adad58a', 'content': 'Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30°C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45°C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.'}, 
    {'UUID': '625ccd1b-d646-4dd2-af2c-4e9ae6047074', 'content': 'The challenge is converting thermal and kinetic energy into electrical signals for the brain. Receptors are key in signal transmission, and they play a massive role here as well.'}, 
    {'UUID': '6b9f21ed-2123-40e2-89ad-fa5ed55d8a87', 'content': 'TRP ion channels and temperature/chemical heat sources\n- TRPV1 receptor (warm receptor) responsible for burning sensation from capsaicin\n- Influx of sodium and calcium into nerve cells\n- Initiates firing of nerves signaling to brain\n- TRPM8 receptors conduct messages about cold stimuli'}, 
    {'UUID': '7079049c-f4d3-47a3-af9c-9c660e801665', 'content': '`"Explaining a topic related to cellular and molecular biology is the goal today. This subject has recently begun to make sense after taking a course in this field."`'}, 
    {'UUID': '775ee194-fe66-439a-b1b5-048f41bb0e58', 'content': '`"Explaining a topic related to cellular and molecular biology is the goal today. This subject has recently begun to make sense after taking a course in this field."`'}
  ], 
  [
    {'UUID': '775ee194-fe66-439a-b1b5-048f41bb0e58', 'content': '`"Explaining a topic related to cellular and molecular biology is the goal today. This subject has recently begun to make sense after taking a course in this field."`'}, 
    {'UUID': '78132fee-d8c0-40d0-ab0d-b813bb90b12b', 'content': 'Sure, let\'s discuss the "burning" sensation! It\'s important to note that this feeling is different from the "hot" sensation we previously covered. A burning sensation can be a sign of irritation or injury to the skin or mucous membranes. It can be caused by a variety of factors, such as exposure to chemicals, extreme temperatures, or certain medical conditions. If you\'re experiencing a burning sensation, it\'s important to identify the cause and seek appropriate treatment to prevent further damage.'}, 
    {'UUID': '79612291-e138-486b-baaa-c3a20b7853db', 'content': 'Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30°C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45°C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.'}, 
    {'UUID': '7ce310f6-fb52-4e99-9485-6ef31f0004e8', 'content': 'Thermoreceptors in the dermis are free nerve endings that extend to the mid-epidermis. They are not enclosed within a membrane, allowing them to detect physical stimuli through interactions with our skin. There are two types of thermoreceptors: cold and warm receptors, which can vary in concentration throughout the body. For example, the ears and face have a greater concentration of cold receptors, making them more susceptible to feeling cold in winter.'}, 
    {'UUID': '7d14e6f1-54d6-4cd3-9946-96e9a377e05b', 'content': 'TRP ion channels and temperature perception | Source\n-------------------------------------------------\n\nTRP ion channels are sensitive to temperature and chemical heat sources, such as capsaicin in chilli peppers. The TRPV1 receptor, also known as the "warm receptor," is responsible for the burning sensation in the mouth caused by capsaicin. This receptor allows an influx of sodium and calcium into nerve cells, initiating the firing of nerves that signal to the brain that something is burning. Similarly, TRPM8 receptors conduct messages about cold stimuli when cooling agents like methanol bind to them. The ability of the body to perceive changes in temperature through these receptors is fascinating, as it demonstrates the body\'s ability to translate physical or chemical stimuli into signals that the brain can understand. For more information on other types of receptors that signal responses to physical stimuli, check out the following short video.'}, 
    {'UUID': '8bfb6c93-9766-4fb5-aa68-cfb5cf7aa58a', 'content': 'Thermoreceptors in the dermis are free nerve endings that extend to the mid-epidermis. They are not enclosed within a membrane, allowing them to detect physical stimuli through interactions with our skin. There are two types of thermoreceptors: cold and warm receptors, which can vary in concentration throughout the body. For example, the ears and face have a greater concentration of cold receptors, making them more prone to feeling cold in winter.'}
  ], 
  [
    {'UUID': '8bfb6c93-9766-4fb5-aa68-cfb5cf7aa58a', 'content': 'Thermoreceptors in the dermis are free nerve endings that extend to the mid-epidermis. They are not enclosed within a membrane, allowing them to detect physical stimuli through interactions with our skin. There are two types of thermoreceptors: cold and warm receptors, which can vary in concentration throughout the body. For example, the ears and face have a greater concentration of cold receptors, making them more prone to feeling cold in winter.'}, 
    {'UUID': '8f038119-5328-4bcf-8c3f-fa6229c31a94', 'content': 'Sure, let\'s discuss the "burning" feeling you mentioned. It\'s possible that you\'re referring to a sensation of heat that is more intense and persistent than the typical warmth we feel. This type of heat can be caused by a variety of factors, such as inflammation, infection, or nerve damage.\n\nInflammation is a natural response of the body to injury or infection, and it can cause a localized increase in temperature, redness, and swelling. This is why we might feel a burning sensation when we have a cut or scrape that becomes infected.\n\nNerve damage can also cause a burning sensation, as damaged nerves may send incorrect signals to the brain. This is known as neuropathic pain, and it can be caused by conditions such as diabetes, shingles, or multiple sclerosis.\n\nRegardless of the cause, a burning sensation can be uncomfortable and even debilitating. If you\'re experiencing this type of heat, it\'s important to speak with a healthcare professional to determine the underlying cause and develop an appropriate treatment plan.'}, 
    {'UUID': '98f1b511-bd03-43d6-93af-be88306cc4e8', 'content': 'Sure, let\'s discuss the "burning" feeling you mentioned. It\'s possible that you\'re referring to a sensation of heat that is more intense and persistent than the typical warmth we feel. This type of heat can be caused by a variety of factors, such as inflammation, infection, or nerve damage.\n\nInflammation is a natural response of the body to injury or infection, and it can cause a localized increase in temperature, redness, and swelling. This is why we might feel a burning sensation when we have a cut or scrape that becomes infected.\n\nNerve damage can also cause a burning sensation, as damaged nerves may send incorrect signals to the brain. This is known as neuropathic pain, and it can be caused by conditions such as diabetes, shingles, or multiple sclerosis.\n\nRegardless of the cause, a burning sensation can be uncomfortable and even debilitating. If you\'re experiencing this type of heat, it\'s important to speak with a healthcare professional to determine the underlying cause and develop an appropriate treatment plan.'}, 
    {'UUID': 'aab58a62-2d9a-47ac-8d1b-e788e4318d35', 'content': "To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on neuron firing frequency. We interact with objects at room temperature, which causes normal firing rates in thermoreceptors. However, cold or hot stimuli alter the firing rate of their corresponding thermoreceptors."}
  ], 
  [
    {'UUID': 'aab58a62-2d9a-47ac-8d1b-e788e4318d35', 'content': "To understand FNE thermoreceptors, you need to know about neurotransmission. The brain's signal strength relies on neuron firing frequency. We interact with objects at room temperature, which causes normal firing rates in thermoreceptors. However, cold or hot stimuli alter the firing rate of their corresponding thermoreceptors."}, 
    {'UUID': 'bd5c4782-2a08-4185-93b2-a9d91bc08ee6', 'content': "Exploring the cellular and molecular mechanisms of temperature sensation goes beyond the average kinetic energy of substances. When you touch something cold, like an ice cube, how does your brain know it's cold? This involves understanding how the measure of kinetic energy is converted into signals that our brain can identify as cold.\n\nAt the molecular level, temperature sensation is mediated by ion channels in specialized nerve endings called nociceptors. These channels are sensitive to changes in temperature and open or close in response to these changes, allowing ions to flow in and out of the cell. This ion flow generates electrical signals that are transmitted to the brain, where they are interpreted as temperature sensations.\n\nOne example of a temperature-sensitive ion channel is the TRPM8 channel, which is activated by cold temperatures. When the temperature drops, the TRPM8 channel opens, allowing calcium ions to flow into the cell. This ion flow generates an electrical signal that is transmitted to the brain, signaling the sensation of cold.\n\nOverall, the cellular and molecular mechanisms of temperature sensation involve the activation of temperature-sensitive ion channels in specialized nerve endings, which convert changes in temperature into electrical signals that are transmitted to the brain."}, 
    {'UUID': 'e38b458b-28e4-4e99-84e4-f8b0c16ad308', 'content': 'Temperature sensation involves intricate cellular and molecular mechanisms. High school chemistry explains the relationship between temperature and the average kinetic energy of molecules, but not how our brains sense and relay this information. The process of interpreting cold, such as when touching an ice cube, involves a series of cellular and molecular events that enable temperature perception.'}, 
    {'UUID': 'e8ace0ea-0f00-4c7c-80a9-de593bcc1289', 'content': 'Perception of temperature is determined by the firing rates of cold and warm receptors. When touching an object between 5-30°C, cold receptor firing increases and warm receptor firing decreases, indicating cold. When touching an object between 30-45°C, warm receptor firing increases and cold receptor firing decreases, indicating warmth. If an object is hot enough to cause pain, nociceptors are activated, signaling pain and further indicating potential injury.'}, 
    {'UUID': 'f00d1e2c-32ff-47ae-b17e-14f8fee12a76', 'content': '`"Explaining a topic related to cellular and molecular biology is the goal today. This subject has recently begun to make sense after taking a course in this field."`'}, 
    {'UUID': 'f93afd2f-fc8e-44c2-9957-458fc837f26e', 'content': 'TRP ion channels are sensitive to temperature and chemical heat sources, such as capsaicin in chilli peppers. The TRPV1 receptor, also known as the "warm receptor," is responsible for the burning sensation in the mouth caused by capsaicin. This receptor allows an influx of sodium and calcium into nerve cells, initiating the firing of nerves that signal to the brain that something is burning. Similarly, TRPM8 receptors conduct messages about cold stimuli when cooling agents like methanol bind to them. The way we perceive changes in temperature is fascinating, as it involves the detection of small movements of molecules. For more information on other physical stimuli receptors, check out this short video.\n\nSource: <https://www.khanacademy.org/science/biology/human-biology/sensory-system/v/sensory-receptors-and-transduction>'}
  ]
]
```

# ERROR PSYCOPG2 CONNECTION
- Need to be careful and close the connection of the db at the end of the workflow an
```bash
psycopg2.InterfaceError: connection already closed
```

- OR need to get rid of the conn parameter and put inside the function the connection to be opened
```bash
conn = connect_db()
```

- OR use a `try/except/finally` and do the whole stuff that needs to be done in the `try` and close the connection in the `finally`
def fetch_documents(table_name: str) -> List[Dict[str, Any]]:
    conn = connect_db()
    try:
        # Fetch data
        # ...
    finally:
        conn.close()

# ollama embeddings
If using `ollama` for embeddings from `langchain.community` you need to find the `ollama.py` file in the virtual env and change the model for the one that you want to use for the embeddings. it is by default set to `llama2`
- location: `/home/creditizens/langgraph/langgraph_venv/lib/python3.10/site-packages/langchain_community`
- filename: `ollama.py`
- approx. line number: `37`
- change to be done: change `model: str = "llama2"` for `model: str = "mistral:7b"`

if you omit that you can check if you want to install ollama and do embedding using its `pip` installation	


# Next
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one.
- test retrieval from data saved by our test.py OK
- create all functions that each nodes will need (tools and other functions)
- create states and use workflow functions to save in each state, name the state the name of the node and the name of the edge fo easy logic and limit confusion, therefore, separate states (have many mini states) and find a way to empty those at the right moment with one function at the end of graph or some depending on the logic.
- See if you need function to get rid of DB info. Create the function that resets the db to zero - OK
- reset the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB.




8b4a37ce-0e43-4b40-aa05-79c09d07e48d | docs/feel_temperature.pdf | 'Decoding ld?: A Cellular & Molecular Perspective'                                                                                                                                                                                                                                                                                                                                  | `"Explaining a topic related to cellular and molecular biology is the goal today. This subject has recently begun to make sense after taking a course in this field."`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

 d9472503-1aaa-41f3-bc16-329617b091e9 | docs/feel_temperature.pdf | "Converting Temperature to Electric Signals: Thermoreceptor Function"                                                                                                                                                                                                                                                                                                                 | The challenge is converting thermal and kinetic energy into electrical signals for the brain. Receptors are key in transmitting signals, and here's no exception. They play a massive role in receiving and transmitting temperature information.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | f

 18237035-48d0-490f-8d23-80f609d932bd | docs/feel_temperature.pdf | "Thermoreceptors in Skin: Free Nerve Endings for Heat & Cold"                                                                                                                                                                                                                                                                                                                         | Thermoreceptors in the dermis are free nerve endings that extend to the mid-epidermis. They are not enclosed within a membrane, allowing them to detect physical stimuli through interactions with our skin. There are two types of thermoreceptors: cold and warm receptors, which can vary in concentration throughout the body. For example, the ears and face have a greater concentration of cold receptors, making them more susceptible to feeling cold in winter.

List[Tuple[Documents, score]]
[
 (
 Document(
   page_content=
     '[
       {"UUID": "124d2bec-5a6d-4ba7-9180-94245e78cfad", "content": "Explore AI tools by Chikara Houses in the Openai GPT store. Improve remote work and well-being with their GPTs. Visit their unique shopping rooms, too."},
       {"UUID": "28902448-443b-41a8-8e35-e6e3cb571fa8", "content": "Explore Chikara\'s social media rooms for shopping tours and AI assistant demos.\\nTrack followers, sales, and enhance your lifestyle with Chikara Houses.\\n[Shop Now](Shop Now!)| [Play & Work with AI](Amazing Chikara AI Assistants!)"
       }
     ]'
   ), 
   0.584969858173907
 ),
 (
 Document(
   page_content=
     '[
       {"UUID": "28902448-443b-41a8-8e35-e6e3cb571fa8", "content": "Explore Chikara\'s social media rooms for shopping tours and AI assistant demos.\\nTrack followers, sales, and enhance your lifestyle with Chikara Houses.\\n[Shop Now](Shop Now!)| [Play & Work with AI](Amazing Chikara AI Assistants!)"},
       {"UUID": "97a4b55f-8c22-4b97-bd20-cc245304b5ac", "content": "Get exclusive access to [Different Collection Rooms](http://www.example.com/collectionrooms) for non-subscribers! Visit the description page to enhance your home specifically. Discover [intriguing stories and articles](http://www.example.com/articles)."
       }
     ]'
   ),
   0.6058349904951864
 )
]


# Next
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one.
- data saved to adb and workflow fine but need now to work on the redis cache retrieval part and vectordb search and falldown to a internet search with a separate func for next agent - OK
- create all functions that each nodes will need (tools and other functions)
- create states and use workflow functions to save in each state, name the state the name of the node and the name of the edge fo easy logic and limit confusion, therefore, separate states (have many mini states) and find a way to empty those at the right moment with one function at the end of graph or some depending on the logic.
- See if you need function to get rid of DB info. Create the function that resets the db to zero - OK
- reset the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB.

# REDIS-CLI

```bash
# from redi-cli, get all keys stored
KEYS *
# or
SCAN 0

# from bash terminal, iterate through keys and get their values `key -> value`
redis-cli --scan --pattern "*" | while read key; do echo "$key -> $(redis-cli get $key)"; done

# delete all keys everythign from the redis server
redis-cli FLUSHALL

# delete keys on a selected redis database only
redis-cli FLUSHDB
```


# PGVECTOR Deprecation new import (https://api.python.langchain.com/en/latest/vectorstores/langchain_postgres.vectorstores.PGVector.html)[docs]
- before: `from langchain_community.vectorstores.pgvector import PGVector`
- now: `from langchain_postgres import PGVector` OR `from langchain_postgres.vectorstores import PGVector`
```bash
# install
pip install -qU langchain-postgres
# run in docker if wanted (we will use our own postgresql datrabase on the server and activate the pgvector extension in this project)
docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16
```


# Next
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one.
- make the tool for internet search fall node which will be the falldown of our query cache/vectordb search fail. This will start next point here do nodes and tools...
- create all functions that each nodes will need (tools and other functions) -OK
- create states and use workflow functions to save in each state, name the state the name of the node and the name of the edge for easy logic and limit confusion, therefore, separate states (have many mini states) and find a way to empty those at the right moment with one function at the end of graph or some depending on the logic.
- reset the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB.


# PROMPTS TEMPLATE AND LLM CALL

### we will use this format of prompt template and will fill it depending of which kind of llm call we need to perform using this format: `<name_of_function>_prompt`
```python
<name_of_function>_prompt = {
  "system": {
    "template": "Answer putting text in markdown tags ```markdown ``` using no more than {maximum} characters to summarize this: {row['text']}.", 
    "input_variables": {}
  },
  "human": {
    "template": "", 
    "input_variables": {}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}
```

### we will use this function to see if we are calling llm using question/answer only or chat version system/human/ai
```python
def make_normal_or_chat_prompt_chain_call(llm_client, prompt_input_variables_part: Dict, prompt_template_part: Optional[Dict], chat_prompt_template: Optional[Dict]):
  
  # default prompt question/answer
  if prompt_template_part:
    prompt = (
      PromptTemplate.from_template(prompt_template_part)
    )
    response = call_chain(llm_client, prompt, prompt_input_variables_part, {})
    return response
  
  # chat prompts question/answer system/human/ai
  elif chat_prompt_template:
    prompt = (
      SystemMessage(content=chat_prompt_template["system"]["template"]) + HumanMessage(content=chat_prompt_template["human"]["template"]) + AIMessage(content=chat_prompt_template["ai"]["template"])
    )
    response = call_chain(llm_client, prompt, {}, {"system": chat_prompt_template["system"]["input_variables"], "human": chat_prompt_template["human"]["input_variables"], "ai": chat_prompt_template["ai"]["input_variables"]})
    return response
  return {'error': "An error occured while trying to create prompts. You must provide either: `prompt_template_part` with `prompt_input_variables_part` or `chat_prompt_template` - in combinaison with llm_client which is always needed." }
```

### Then the llm calling function will call the llm in different way (formatting the call as needed)
```python
def call_chain(model: ChatGroq, prompt: PromptTemplate, prompt_input_vars: Optional[Dict], prompt_chat_input_vars: Optional[Dict]):
  # only to concatenate all input vars for the system/human/ai chat
  chat_input_vars_dict = {}
  for k, v in prompt_chat_input_vars.items():
    for key, value in v.items():
      chat_input_vars_dict[key] = value
  print("Chat input vars: ",  chat_input_vars_dict)

  # default question/answer
  if prompt and prompt_input_vars:
    print("Input variables found: ", prompt_input_vars)
    chain = ( prompt | model )
    response = chain.invoke(prompt_input_vars)
    print("Response: ", response, type(response))
    return response.content.split("```")[1].strip("markdown").strip()
    
  # special for chat system/human/ai
  elif prompt and prompt_chat_input_vars:
    print("Chat input variables found: ", prompt_chat_input_vars)
    chain = ( prompt | model )
    response = chain.invoke(chat_input_vars_dict)
    print("Response: ", response, type(response))
    return response.content.split("```")[1].strip("markdown").strip()
```
**Step of the prompt template use for llm call:**
.1 import the right prompt template normal question/response or chat special system/human/aione
.2 use the template and the function argument to create the prompt
.3 use make_normal_or_chat_prompt_chain_call(llm_client, prompt_input_variables_part: Dict, prompt_template_part: Optional[Dict], chat_prompt_template: Optional[Dict])
.4 use internet search tool if nothing is found or go to the internet search node

Eg.:
```python
from prompts.prompts import test_prompt_tokyo, test_prompt_siberia, test_prompt_monaco, test_prompt_dakar

print("SIBERIA: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, test_prompt_siberia["input_variables"], test_prompt_siberia["template"], {}))
time.sleep(0.5)
print("TOKYO: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, test_prompt_tokyo["input_variables"], test_prompt_tokyo["template"], {}))
time.sleep(0.5)
print("DAKAR: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, {}, {}, test_prompt_dakar))
time.sleep(0.5)
print("MONACO: \n", make_normal_or_chat_prompt_chain_call(groq_llm_llama3_8b, {}, {}, test_prompt_monaco))

'''
# prompts that also fit the function as question/answer but we have decided to use to next example one to have same template and use or fill what is needed from it
test_prompt_siberia = {"template": "What is the weather like in {test_var} usually.\nAnswer only with the schema in markdown between ```markdown ```.", "input_variables": {"test_var": "Siberia"}}
test_prompt_tokyo = {"template": "We is the weather like in Japan in summer usually.\nAnswer only with the schema in markdown between ```markdown ```.", "input_variables": {}}

# Prompt System/Human/Ai that will be used from now on for all prompts
test_prompt_dakar = {
  "system": {
    "template": "You are an expert in climat change and will help by answering questions only about climat change. If any other subject is mentioned, you must answer that you don't know as you are only expert in climat change related questions.\nAnswer only with the schema in markdown between ```markdown ```.", 
    "input_variables": {}
  },
  "human": {
    "template": "I need two know if Dakar is going to struggle in the future beacause of the rise of ocean water level due to global warming? {format_of_answer}", 
    "input_variables": {"format_of_answer": "Format your answer please using only one sentence and three extra bullet points."}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}
test_prompt_monaco = {
  "system": {
    "template": "You are an expert in climat change and will help by answering questions only about climat change. If any other subject is mentioned, you must answer that you don't know as you are only expert in climat change related questions.\nAnswer only with the schema in markdown between ```markdown ```.", 
    "input_variables": {}
  },
  "human": {
    "template": "I am following F1 and want to know when is the next Monaco GP and will {car_house} win again?", 
    "input_variables": {"car_house": "Ferrari"}
  },
  "ai": {
    "template": "I believe that Renault will win next {city} GP", 
    "input_variables": {"city": "Singapour"}
  },
}
'''
```

**Now We have a prompt template and will use the same and will call llms using this kind of example if AI message is needed if will be added but in this example we use just human and system message**

##### prompt template
```code
<function_name>_prompt = {
  "system": {
    "template": """
      <system_message_template_with_or_without_input_variables>
      """, 
    "input_variables": {}
  },
  "human": {
    "template": "", 
    "input_variables": {}
  },
  "ai": {
    "template": "", 
    "input_variables": {}
  },
}
```

##### formatting llm call taking what is needed from the imported prompt template
```python
system_message_tuple = ("system",) + (prompt["system"]["template"],)
human_message_tuple = ("human",) + (query.strip(),)
messages = [system_message_tuple, human_message_tuple]

llm_called = llm.invoke(messages)

llm_called_answer = llm_called.content.split("```")[1].strip("markdown").strip()
```



### list of function for node or tool:
#### 1) User initial query management and files or url if exist
- process_query: returns tuple (df_final, content_dict) : will decompose user query and create df of url or pdf OR returns Dict[str,str] {message: user query text to answer to as no url/pdf processing needed}
state saved: question/pdf or question/url or text and webpage or url df

- store_dataframe_to_db: returns Dict[str, str] {error/success: message}
state saved: df saved bool True/False
custom_chunk_and_embed_to_vectordb: returns Dict[str, str] {error/success: message}
state saved: db data chunked and embedded bool True/False

#### 2) Query Retrieval
- query_redis_cache_then_vecotrdb_if_no_cache: returns List[Dict[str,Any]] | str | Dict[str,str] -->
  - List[Dict[str,Any]] is found answer cache/vectordb; 
  - str if nothing found;
  - Dict[str,str] if error {error:message}
state saved: retrieved redis or vectordb -->
  - query hash retrieved List[DICT[str,Any]]
  - query vector retrieved List[DICT[str,Any]]
  - vectordb retrieved List[DICT[str,Any]]
  - nothing retrieved: bool
 
#### Extras
- delete_table
- subprocess function to run code safely in docker tools


### Format of how to make tools
# tool template
class <name_arg_template>(BaseModel):
      <arg_name>: <input_type> = Field(default=<default_value>, description="<description>")

def <func_name>(<arg_name>: <input_type> = <default_value_if_any>) -> <return_type>:
  """
    <func description> 
    
    Parameter: 
    <arg_name> <input_type> : '<descrption>' = {<default_value_if_any>}
    
    Returns: 
    <return_type> <description> 
  """
  <func_logic>
  return <returned_object_stuff>

<var_tool_name> = StructuredTool.from_function(
  func=<func_name>,
  name="<name_of_tool>",
  description=  """
    <description_of_tool>
  """,
  args_schema=<name_arg_template>,
  # return_direct=True, # returns tool output only if no TollException raised
  # coroutine= ... <- you can specify an async method if desired as well
  # callback==callback_function # will run after task is completed
)


# Visualize graph
```python
# install
sudo apt-get install graphviz graphviz-dev
pip install pygraphviz
# install and display graph command
pip install ipython
display(Image(app.get_graph().draw_png()))
```


# Agent Tools Workflow

Just use @tool decorator on any function and use ToolNode.
Agent node which has binded tools will choose which tool is good to use and the next node ToolNode will just execute the tool choosen by the Agent which has already field the schema with the query and have the right schema for that tool to be executed by the ToolNode.
Then save returned values to the state for easy graph workflow.

```bash
# query
Do you need any help? any PDF doc or webpage to analyze? please search online about the latest restaurant opened in Azabu Juuban?

# Step 1
{'get_user_input': {'messages': ['please search online about the latest restaurant opened in Azabu Juuban?']}}

# Step 2
{'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_m6a0', 'function': {'arguments': '{"query": "latest restaurant opened in Azabu Juuban"}', 'name': 'search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 34, 'prompt_tokens': 3180, 'total_tokens': 3214, 'completion_time': 0.11005241, 'prompt_time': 0.984584956, 'queue_time': 0.0012828750000000166, 'total_time': 1.094637366}, 'model_name': 'llama3-groq-70b-8192-tool-use-preview', 'system_fingerprint': 'fp_ee4b521143', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-77b04c0e-c49c-437f-9257-31ccd94ce4e8-0', tool_calls=[{'name': 'search', 'args': {'query': 'latest restaurant opened in Azabu Juuban'}, 'id': 'call_m6a0', 'type': 'tool_call'}], usage_metadata={'input_tokens': 3180, 'output_tokens': 34, 'total_tokens': 3214})]}}

# Step 3
{'tool_search_node': {'messages': [ToolMessage(content='{"messages": ["Fukasaka is a sushi restaurant that opened in March 2024, led by a chef who trained at renowned establishments like the Michelin-starred Sushi Hashimoto in Shintomicho. ... Azabu-juban Station Directions from station 5 min. walk from Azabu-juban Station Exit 1 ... Aoyama is a vibrant district where you can experience the latest trends, unique ... A main dish of yum pam, phu penang curry, kai chao mu sap or muta kai tote will satisfy the palate. This eatery also offers a course menu with desserts. Blue Papaya Thailand is located near the Tokyo Metro Nanboku Line Azabu-Juban Station. It provides dinner and lunch services Monday through Saturday. The world-renowned restaurant, which features cuisine that is uncommon in Japan, opened its doors with a menu developed by Ichiro Ozaki, the owner-chef of Azabu Juban Ozaki, a popular sushi kappo restaurant located on the first floor of the same building, to offer new Japanese-style vegan dishes. The restaurant is a 5-minute walk from Azabu-Juban Station and offers a stylish atmosphere with 19 seats available. The opening hours are from Tuesday to Sunday, 17:30 to 23:30, with the last order for food at 22:00 and for drinks at 22:30. The restaurant is closed on certain days of the month, specifically the first and third Sundays. At this ... The restaurant \\"Principio\\" is located in Azabu-Juban, Minato-ku, Tokyo, and is accessible on foot from Azabu-Juban Station. The interior of the restaurant has a cozy and tranquil atmosphere, with 5 tables and a total of 10 seats available."]}', name='search', tool_call_id='call_m6a0')]}}

# intermediary print of the state messages
Message state:  [HumanMessage(content='message initialization', id='dba33398-1603-4a31-8650-5ba76542f1aa'), HumanMessage(content='please search online about the latest restaurant opened in Azabu Juuban?', id='5b451ab8-c184-462e-beba-391e82701732'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_m6a0', 'function': {'arguments': '{"query": "latest restaurant opened in Azabu Juuban"}', 'name': 'search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 34, 'prompt_tokens': 3180, 'total_tokens': 3214, 'completion_time': 0.11005241, 'prompt_time': 0.984584956, 'queue_time': 0.0012828750000000166, 'total_time': 1.094637366}, 'model_name': 'llama3-groq-70b-8192-tool-use-preview', 'system_fingerprint': 'fp_ee4b521143', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-77b04c0e-c49c-437f-9257-31ccd94ce4e8-0', tool_calls=[{'name': 'search', 'args': {'query': 'latest restaurant opened in Azabu Juuban'}, 'id': 'call_m6a0', 'type': 'tool_call'}], usage_metadata={'input_tokens': 3180, 'output_tokens': 34, 'total_tokens': 3214}), ToolMessage(content='{"messages": ["Fukasaka is a sushi restaurant that opened in March 2024, led by a chef who trained at renowned establishments like the Michelin-starred Sushi Hashimoto in Shintomicho. ... Azabu-juban Station Directions from station 5 min. walk from Azabu-juban Station Exit 1 ... Aoyama is a vibrant district where you can experience the latest trends, unique ... A main dish of yum pam, phu penang curry, kai chao mu sap or muta kai tote will satisfy the palate. This eatery also offers a course menu with desserts. Blue Papaya Thailand is located near the Tokyo Metro Nanboku Line Azabu-Juban Station. It provides dinner and lunch services Monday through Saturday. The world-renowned restaurant, which features cuisine that is uncommon in Japan, opened its doors with a menu developed by Ichiro Ozaki, the owner-chef of Azabu Juban Ozaki, a popular sushi kappo restaurant located on the first floor of the same building, to offer new Japanese-style vegan dishes. The restaurant is a 5-minute walk from Azabu-Juban Station and offers a stylish atmosphere with 19 seats available. The opening hours are from Tuesday to Sunday, 17:30 to 23:30, with the last order for food at 22:00 and for drinks at 22:30. The restaurant is closed on certain days of the month, specifically the first and third Sundays. At this ... The restaurant \\"Principio\\" is located in Azabu-Juban, Minato-ku, Tokyo, and is accessible on foot from Azabu-Juban Station. The interior of the restaurant has a cozy and tranquil atmosphere, with 5 tables and a total of 10 seats available."]}', name='search', id='74eb112e-76a8-428a-be4a-37bb9aed685b', tool_call_id='call_m6a0')]

# Step 4
{'answer_user': {'messages': [{'role': 'ai', 'content': '{"messages": ["Fukasaka is a sushi restaurant that opened in March 2024, led by a chef who trained at renowned establishments like the Michelin-starred Sushi Hashimoto in Shintomicho. ... Azabu-juban Station Directions from station 5 min. walk from Azabu-juban Station Exit 1 ... Aoyama is a vibrant district where you can experience the latest trends, unique ... A main dish of yum pam, phu penang curry, kai chao mu sap or muta kai tote will satisfy the palate. This eatery also offers a course menu with desserts. Blue Papaya Thailand is located near the Tokyo Metro Nanboku Line Azabu-Juban Station. It provides dinner and lunch services Monday through Saturday. The world-renowned restaurant, which features cuisine that is uncommon in Japan, opened its doors with a menu developed by Ichiro Ozaki, the owner-chef of Azabu Juban Ozaki, a popular sushi kappo restaurant located on the first floor of the same building, to offer new Japanese-style vegan dishes. The restaurant is a 5-minute walk from Azabu-Juban Station and offers a stylish atmosphere with 19 seats available. The opening hours are from Tuesday to Sunday, 17:30 to 23:30, with the last order for food at 22:00 and for drinks at 22:30. The restaurant is closed on certain days of the month, specifically the first and third Sundays. At this ... The restaurant \\"Principio\\" is located in Azabu-Juban, Minato-ku, Tokyo, and is accessible on foot from Azabu-Juban Station. The interior of the restaurant has a cozy and tranquil atmosphere, with 5 tables and a total of 10 seats available."]}'}]}}

```

- Step 1 Output: The user's query ("please search online about the latest restaurant opened in Azabu Juuban?") is captured and added to the MessagesState.

- Step 2 Output: The agent node invokes the LLM with the query. The LLM identifies that it should use the search tool to handle this query, so it generates a tool call with the appropriate arguments (e.g., {'query': 'latest restaurant opened in Azabu Juuban'}). This step is crucial because it shows that the LLM successfully recognized that the search tool should be used and passed the correct query string to it.

- Step 3 Output: The tool_search_node executes the search function using the query provided by the LLM. The search tool performs the internet search and returns the results.

- Step 4 Output: The answer_user node takes the result of the internet search and prepares it for display to the user.


eg. of tools:
```python
groq_llm_mixtral_7b = ChatGroq(
    temperature=float(os.getenv("GROQ_TEMPERATURE")),
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-groq-70b-8192-tool-use-preview",
    max_tokens=int(os.getenv("GROQ_MAX_TOKEN")),
)

# some other tools defined
tool
def get_proverb(query: str, state: MessagesState = MessagesState()):
  """Will transform user query into a funny proverb"""
  system_message = SystemMessage(content="You are an expert in creating funny short proverbs from any query.")
  human_message = HumanMessage(content=f"Create a funny proverb please: {query}")
  response = groq_llm_mixtral_7b.invoke([system_message, human_message])
  return {"messages": [response]}

@tool
def find_link_story(query: str, state: MessagesState = MessagesState()):
  """Will find a link between user query and the planet Mars"""
  system_message = SystemMessage(content="You are an expert in finding links between Mars planet and any query.")
  human_message = HumanMessage(content=f"{query}")
  response = groq_llm_mixtral_7b.invoke([system_message, human_message])
  return {"messages": [response]}


# Define the search tool
internet_search_tool = DuckDuckGoSearchRun()
tool_internet = Tool(
    name="duckduckgo_search",
    description="Search DuckDuckGO for recent results.",
    func=internet_search_tool.run,
)

@tool
def search(query: str, state: MessagesState = MessagesState()):
    """Call to surf the web."""
    search_results = internet_search_tool.run(query)
    return {"messages": [search_results]}

tool_search_node = ToolNode([get_proverb, find_link_story])

# Bind the tool to the LLM
llm_with_internet_search_tool = groq_llm_mixtral_7b.bind_tools([get_proverb, find_link_story])
```

# Function to have graph output in `.stream()` mode beautified
```python
# function to beautify output for an ease of human creditizens reading
def message_to_dict(message):
    if isinstance(message, (AIMessage, HumanMessage, SystemMessage, ToolMessage)):
        return {
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
            "response_metadata": message.response_metadata if hasattr(message, 'response_metadata') else None,
            "tool_calls": message.tool_calls if hasattr(message, 'tool_calls') else None,
            "usage_metadata": message.usage_metadata if hasattr(message, 'usage_metadata') else None,
            "id": message.id,
            "role": getattr(message, 'role', None),
        }
    return message

def convert_to_serializable(data):
    if isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, (AIMessage, HumanMessage, SystemMessage, ToolMessage)):
        return message_to_dict(data)
    return data

def beautify_output(data):
    serializable_data = convert_to_serializable(data)
    return json.dumps(serializable_data, indent=4)

# Example of how to use this function
count = 0
for step in app.stream(
    {"messages": [HumanMessage(content="message initialization")]},
    config={"configurable": {"thread_id": 42}}):
    count += 1
    if "messages" in step:
        print(f"Step {count}: {beautify_output(step['messages'][-1].content)}")
    else:
        print(f"Step {count}: {beautify_output(step)}")
```


# Hypothetical Workflow Scenario
Here we will create hypothetical desired workflow for the graph
```python
"""
--- START 
    0--- ask user query input
    1--- Analyze User Query To Get the Webpage or Document or just text (to know if we need to perform webpage parsing or pdf doc parsing) 
         --- save to state 'webpage' or 'pdf' or 'text'
        1.1--- If/Else conditional edge 'db_doc_check':
           --- output = 'webpage/pdf title IN db'
           1.1.1--- If webpage title or pdf title in db --- If/Else conditional edge 'cache/vectordb query answer check'
                                                        1.1.1.1--- If answer in cache/vectordb --- answer user query --- END
                                                        1.1.1.2--- Else Internet Search about subject
                                                                   --- Embed query/internet_answer and cache it
                                                                       --- answer user query with internet_search result
                                                                           --- END
           --- output = 'webpage/pdf title NOT IN db'
           1.1.2--- Else --- If/Else conditional edge 'webpage or pdf processing':
                         --- output = 'url'
                         1.1.2.1--- If webpage --- parse webpage
                                                   --- store to database
                                                       --- create chunks from database rows
                                                           --- embed chunks
                                                               --- save to state 'webpage embedded'
                                                                   --- go to 1.1.1.1 flow which will search in cache and won't found it and make a vector search and answer-> END
                         --- output = 'pdf'
                         1.1.2.2--- If pdf     --- parse pdf
                                                   --- store to database
                                                       --- create chunks from database rows
                                                           --- embed chunks
                                                               --- save to state 'pdf embedded'
                                                                   --- go to 1.1.1.1 flow which will search in cache and won't found it and make a vector search and answer-> END
                         --- output = 'text'
                         1.1.2.3--- If text    --- perform internet search to get response of `text only` query
                                                   --- format internet result and answer -> END

    2--- Analyze User Query to extract the question from it and rephrase the question to optimize llm information research/retrieval --- save to state 'query/question'
         3--- Retrieve answer from Query in embedded collection
              --- save to state answer_retrieved
                  4--- Internet Search user query
                       --- save to state internet search_answer
                           5--- answer to user in markdown report format using both states answer_retrieved/search_answer
"""
```

# Next
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one.
- incorporate the internet tool in the graph - OK
- graph accepts x3 states but we will only use here `MessagesStates` already present in LangGraph library. We could create other states using `NewState(TypeDict)` but we will keep it simple in this project - MAYBE BUT NOT NOW
- delete `parquet` files (`df_final` compacted) after they have been processed and saved to db to save space and not have file building up and taking space - OK
- reset the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB.
- fix redis that doesn't save anything as value for key. find in the code where is the issue


# Test Internet Search Implementation and markdown report generation

Initial query: what is the first result when we search for 'sakura flower' though the internet?

**The Beauty of Sakura: A Report on Japanese Cherry Blossom Trees and Their Culture**
=====================================================

### Introduction
Sakura, the Japanese word for cherry blossom, is a flower that has captivated the hearts of many around the world. With its delicate petals and vibrant colors, it's no wonder why cherry blossom trees have become a symbol of Japan's rich culture and history. In this report, we will delve into the different varieties of cherry blossom trees, their lifespan, and the cultural significance of hanami, or cherry blossom viewing.


### Types of Cherry Blossom Trees

There are many different varieties of cherry blossom trees, with flowers that range in color from white to deep pink. Most sakura trees are small to medium-sized, with a lifespan of 15-30 years, though some can live longer. Their bark is smooth and reddish-brown, and their green leaves turn vibrant shades of yellow, red, or crimson in autumn.\n\n### Hanami Culture

Hanami, which translates to viewing flowers, is a centuries-old tradition in Japan where people gather under the Sakura tree for picnics and parties. This tradition is a testament to the unifying power of nature and the beauty of cherry blossoms. In Japan, hanami is a time for people to come together and appreciate the fleeting beauty of the cherry blossoms.

### History and Culture of Sakura\n\nSakura is not only a symbol of Japan's natural beauty but also a representation of its rich culture and history. The National Cherry Blossom Festival in Washington, D.C. is a testament to the unifying power of cherry blossoms, and it has become a popular event around the world. Cherry blossom festivals have become increasingly popular outside of Japan, with many countries hosting their own sakura festivals.

### Conclusion
In conclusion, the beauty of sakura is not just about the flower itself, but about the culture and history that surrounds it. Whether you're in Japan or around the world, cherry blossom trees are a symbol of the fleeting nature of life and the importance of appreciating the beauty that surrounds us. So, take a moment to stop and smell the sakura!

### Advice

* If you're planning to visit Japan during cherry blossom season, make sure to book your accommodations and transportation in advance, as it's a popular time to visit.
* Take a moment to appreciate the beauty of the cherry blossoms, and don't forget to take plenty of photos to capture the moment.
* If you're unable to visit Japan, consider attending a cherry blossom festival in your local area, or planting your own cherry blossom tree to enjoy the beauty of the flowers in your own backyard.



# Binding Tools To LLM and Structured Output

Create a class using `BaseModel` and define variables, types of those variables and use also the `Field` to add descriptions.

```python
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_pydantic_to_openai_function
from langchain_core.pydantic_v1 import BaseModel, Field

class InternetSearchStructuredOutput(BaseModel):
    """Internet research Output Response"""
    internet_search_answer: str = Field(description="The result of the internet research in markdon format about user query when an answer has been found.")
    source: str = Field(description="The source were the answer has been fetched from in markdown format, it can be a document name, or an url")
    error: str = Field(description="An error messages when the search engine didn't return any response or no valid answer.")

functions = [format_tool_to_openai_function(tool) for tool in tools]
functions.append(convert_pydantic_to-openai_function(InternetSearchStructuredOutput))
model.bind_functions(function)
result = model.invoke(
  {
    "messages": [HumanMessage(content="Search in internet about Chikarahouses.com, I need some more information")]
  }
)
```

but didn't worked!

the next example didn't work also but we have a kind of a boilerplate the the right output format but nothing if filled. Just empty
```python
from langchain.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# Define structured output schema using Pydantic's BaseModel
class InternetSearchStructuredOutput(BaseModel):
    internet_search_answer: str = Field(description="The result of the search in markdown format.")
    source: str = Field(description="The source of the information.")
    error: str = Field(description="Error if any occurred during the search.")

# Initialize the Groq model and set it to return structured output
model = ChatGroq(model="mixtral-7b").with_structured_output(InternetSearchStructuredOutput)

# Create a query asking for structured output directly
query = HumanMessage(content="Provide information about the next satellite launch in the following structure: \
'internet_search_answer', 'source', and 'error'.")

# Invoke the model and get the structured output
try:
    response = model.invoke([query])
    print("Structured Output:", response)
except Exception as e:
    print(f"An error occurred: {e}")
```
Outputs:

```python
Structured Output: internet_search_answer='' source='' error=''
```

# Structured Output Using Only Prompting Which Has Worked Fine
Trying without the structured_output s***, just prompting as I did before and it worked fine without the complexity of using something that is documented in 1000 different ways and not one working, even chatgpt is tired!
```python
# Initialize the Groq model and set it to return structured output
model = groq_llm_mixtral_7b

# Create a query asking for structured output directly
query_system = SystemMessage(content="Help user by answering always with 'Advice' (your advice about the subject of the question and how user should tackle it), 'Answer' (The answer in French to the user query) , and 'error' (The status when you can't answer), put it in a dictionary between mardown tags like ```markdown{Advice: your advice ,Answer: the answer to the user query in French,error:if you can't answer}```.")
query_human = HumanMessage(content="What is the story behind th epythagorus theorem, it seems like they have lied and it has been stollen knowledge from the Egyption where pythagore have studied with black people and came back to greece and said that it is from his own knwoledge, today french people are teaching that it is greek when it was black egyptian knowledge. this to keep the white supremacy idea which is a bad ideology")

# Invoke the model and get the structured output
try:
    response = model.invoke([query_system, query_human])
    print("Structured Output:", response)
except Exception as e:
    print(f"An error occurred: {e}")
```

That have worked, therefore, we just need to prompt well no need al those crazy stuff and it is very consistant with this mini llm 7b mistral therefore the bigger ones paid one will work fine, this is our assumption and that is why we are doing it all using small llms
Outputs:
```python
Structured Output: content="```markdown{\nAdvice: It's important to approach history with an open mind and a critical eye. While it's true that Pythagoras studied in Egypt, there is no concrete evidence to support the claim that he stole the theorem from Egyptian or black mathematicians. The development of mathematical knowledge is often a collaborative process, with ideas and concepts building upon each other over time. It's crucial to give credit where it's due, but it's also important to avoid oversimplifying or distorting historical narratives to fit a particular ideology.\nAnswer: Le théorème de Pythagore est un théorème de géométrie dans un triangle rectangle, qui affirme que le carré de la longueur de l'hypoténuse (le côté opposé à l'angle droit) est égal à la somme des carrés des longueurs des deux autres côtés. Ce théorème est souvent attribué à Pythagore, un mathématicien grec du VIe siècle avant J.-C., mais il est possible que des versions de ce théorème aient été connues et utilisées par d'autres cultures avant lui.\nError: There is no error in your question, but it's important to note that the development of mathematical knowledge is a complex and nuanced process, and attributing specific discoveries to particular individuals or cultures can be challenging and sometimes contentious.\n}```" response_metadata={'token_usage': {'completion_tokens': 319, 'prompt_tokens': 202, 'total_tokens': 521, 'completion_time': 0.520485911, 'prompt_time': 0.011388187, 'queue_time': 0.003033524000000001, 'total_time': 0.531874098}, 'model_name': 'mixtral-8x7b-32768', 'system_fingerprint': 'fp_c5f20b5bb1', 'finish_reason': 'stop', 'logprobs': None} id='run-19184a14-1ff9-404c-9721-687fd9f81320-0' usage_metadata={'input_tokens': 202, 'output_tokens': 319, 'total_tokens': 521}
```
- adapted and transformed as funciton which can be used as `node` function or as a `@tool`, need to create the prompts
```python
def structured_output_report(state: MessagesState):
  query_system = SystemMessage(content="Help user by answering always with 'Advice' (your advice about the subject of the question and how user should tackle it), 'Answer' (The answer in French to the user query) , and 'error' (The status when you can't answer), put it in a dictionary between mardown tags like ```markdown{Advice: your advice ,Answer: the answer to the user query in French,error:if you can't answer}```.")
  query_human = HumanMessage(content="What is the story behind th epythagorus theorem, it seems like they have lied and it has been stollen knowledge from the Egyption where pythagore have studied with black people and came back to greece and said that it is from his own knwoledge, today french people are teaching that it is greek when it was black egyptian knowledge. this to keep the white supremacy idea which is a bad ideology")

  # Invoke the model and get the structured output
  try:
    response = groq_llm_llama3_8b.invoke([query_system, query_human])
    print(f"structured_output:{response}")
  except Exception as e:
    return f"An error occurred: {e}"

  return {"messages": [response]}
```

```python
@tool
def structured_output_report(prompt: str = structured_outpout_report_prompt, state: MessagesState):
  """
  This function will structure the output in order to have the format of answer required for quality deliverability
  
  Parameters:
  prompt str: the system prompt that will instruct in how the output of the answer should be structured
  
  Returns:
  messages dict: a dictionary that will update the state with the reponse from the llm having structure the message
  """
  messages = state["messages"]
  last_message = messages[-1].content

  query_system = prompt
  query_human = HumanMessage(content=last_message))

  # Invoke the model and get the structured output
  try:
    response = groq_llm_llama3_8b.invoke([query_system, query_human])
    print(f"structured_output:{response}")
  except Exception as e:
    return f"An error occurred: {e}"

  return {"messages": [response]}

tool_search_node = ToolNode([structured_output_report])

# tool_choice="any" only supported for the moment for MistraiAI, Openai, Groq, FireworksAI. for grow should ane `None` or `auto`
llm_with_structured_output_report_tool = groq_llm_mixtral_7b.bind_tools([structured_output_report])
```

# Structure Outputs all different ways and feedback
- Define your desired data structure.
```python
class AddCount(BaseModel):
    """Will always add 10 to any number"""
    initialnumber: List[int] = Field(default= [], description="Initial numbers present in the prompt.")
    calculation: str = Field(default="", description="Details of the calculation that have create the 'result' number.")
    result: int = Field(default= 0, description="Get the sum of the all the numbers. Then add 10 to the sum which will be final result. ")
    error: str = Field(default= "", description="An error message when no number has been found in the prompt.")

class CreateBulletPoints(BaseModel):
    """Creates answers in the form of 3 bullet points."""
    bulletpoints: str = Field(default="", description="Answer creating three bullet points that are very pertinent.")
    typeofquery: str = Field(default="", description="tell if the query is just a sentence with the word 'sentence' or a question with the word 'question'.")
```

- **First way**
```python
# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=CreateBulletPoints)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# And a query intended to prompt a language model to populate the data structure.
prompt_and_model = prompt | groq_llm_mixtral_7b
output = prompt_and_model.invoke({"query": "Tokyo is number 1 city to have more than 20 million habitants. Read description of tool and provide right answer for each fields please."})
response = parser.invoke(output)
print("First way: ", response)

Outputs condensed bullet points but has respected the schema and in case and in case of calculaiton does it right:
First way:  bulletpoints='• Tokyo has the largest population of any city, with over 20 million inhabitants.\n• It is a major international financial and cultural center.\n• Tokyo is known for its efficient public transportation system and high standard of living.' typeofquery='sentence'
First way:  initialnumber=[1, 20000000] calculation='Summed the initial number and added 10' result=2000010 error=''

```

- **Second way**
```python
parser = PydanticOutputFunctionsParser(pydantic_schema=CreateBulletPoints)

openai_functions = [convert_to_openai_function(CreateBulletPoints)]
chain = prompt | groq_llm_mixtral_7b.bind(functions=openai_functions) | parser

print("Second way: ", chain.invoke({"query": "Tokyo is number 1 city to have more than 20 million habitants. Read description of tool and provide right answer for each fields please."}))

Outputs no bullet points and does not the calculation at all, miss interprets:
Second way:  bulletpoints='Tokyo is the largest city in Japan and one of the 14 cities in the world with a population of more than 20 million people. It is a major international financial and cultural center. Tokyo has a population of over 37 million people, making it the most populous metropolitan area in the world.' typeofquery='sentence'
Second way:  initialnumber=[20000000, 1] calculation='Added initialnumber values to get sum' result=0 error=''

```

- **Third way**
```python
# for AddCount
#query_system = SystemMessage(content="Help user by answering always with 'initialnumber' (Initial numbers present in the prompt.), 'calculation' (Details of the calculation that have create the 'result' number), 'result' (Get the sum of the all the numbers. Then add 10 to the sum which will be final result) and 'error' {An error message when no number has been found in the prompt.), put it in a dictionary between mardown tags like ```markdown{initialnumber: list of Initial numbers present in the prompt. ,calculation: Details of the calculation that have create the 'result' number, result: Get the sum of the all the numbers. Then add 10 to the sum which will be final result, error: An error message when no number has been found in the prompt.}```.")
# for CreateBulletPoints
query_system = SystemMessage(content="Help user by answering always with 'bulletpoints' (Answer creating three bullet points that are very pertinent.), 'typeofquery' (tell if the query is just a sentence with the word 'sentence' or a question with the word 'question'.), put it in a dictionary between mardown tags like ```markdown{bulletpoints: Answer creating three bullet points that are very pertinent, typeofquery: tell if the query is just a sentence with the word 'sentence' or a question with the word 'question'.}```.")
query_human = HumanMessage(content="Tokyo is number 1 city to have more than 20 million habitants. Read description of tool and provide right answer for each fields please.")

# Invoke the model and get the structured output
response = groq_llm_mixtral_7b.invoke([query_system, query_human])
print(f"Third way: {response.content.split('```')[1].strip('markdown').strip()}")

Outputs th ebets structure and follows instruction well with nice markdown formatting and in case of call calculations sees `20 million` as `20` and not `20_000_000`:
Third way: {
bulletpoints: 
- Tokyo is the most populous city in the world with over 20 million inhabitants.
- It is the capital of Japan and is known for its bustling streets, rich culture, and technological advancements.
- Tokyo is a popular tourist destination, attracting millions of visitors each year with its many attractions, including the Tokyo Tower, the Tokyo Skytree, and the historic Asakusa district.
typeofquery: sentence
}
Third way: {
initialnumber: [20, 1], 
calculation: The sum of the two initial numbers is 21. Then add 10 to the sum which gives a final result of 31.
result: 31, 
error: There are initial numbers present in the prompt, so no error message is needed.
}
```

- **Fourth way**
```python
model = groq_llm_mixtral_7b.bind_tools([CreateBulletPoints])
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are helpful assistant"), ("user", "{input}")]
)
parser = JsonOutputToolsParser()
chain = prompt | model | parser
response = chain.invoke({"input": "Tokyo is number 1 city to have more than 20 million habitants. Read description of tool and provide right answer for each fields please."})
print(f"Fourth way: {response}")

Outputs no bullet points and in case of calculation doesn't do it right:
Fourth way: [{'args': {'bulletpoints': 'Tokyo is the number 1 city with more than 20 million inhabitants, it is known for its rich culture and history, and it has a vibrant economy.', 'typeofquery': 'statement'}, 'type': 'CreateBulletPoints'}]
Fourth way: [{'args': {'calculation': 'Adding 10 to the population of Tokyo (20 million) to find the new total.', 'initialnumber': [20000000], 'result': 0}, 'type': 'AddCount'}]

```

- **Fifth way**
```python
response_schemas = [
    ResponseSchema(name="bulletpoints", description="Answer creating three bullet points that are very pertinent."),
    ResponseSchema(name="typeofquery", description="tell if the query is just a sentence with the word 'sentence' or a question with the word 'question'."),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="answer the user query.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)
chain = prompt | model | output_parser
response = chain.invoke({"query": "Tokyo is number 1 city to have more than 20 million habitants. Read description of tool and provide right answer for each fields please."})
print(f"Fith way: {response}")
#print("Fith Way with parser: ", output_parser.invoke(response))
```

- **Feedback on output parsers**
Best is the **First way** and the **Third way**.
I will be using probably the **Third way** as it is more consistent and the output is a str and we can use a function that we already have to transform it into a dict and fetch what we need.
The Fifth way didn't work always an error:
```python
...
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
...
    raise OutputParserException(f"Got invalid JSON object. Error: {e}")
langchain_core.exceptions.OutputParserException: Got invalid JSON object. Error: Expecting value: line 1 column 1 (char 0)
```



# Next
- keep in mind the pdf parser that might need to be refactored to chunk using same process as the webpage parser one.
- reset the cache to zero as well so that we have the option to delete everything for some future task that doesn't need the data to persist forever in the DB.
- fix redis that doesn't save anything as value for key. find in the code where is the issue





