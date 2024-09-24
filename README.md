# LangGraph

### Idea behind this project and objectives
I believe that all is about the data embedded and how relevant it can be and believe that difficult structured docs like pdf can't be fully embedded btu can be  quality-embedded extracting and storing only the information that can be extracted in a qualitative way and then use internet search if retriever didn't find the information or found the information not complete.

Therefore, here i want highly customizable agents framework like LangGraph and need those edges and conditions and nodes to be able to create an advanced LLM task support.

I also believe in mini-agents making super-specific tasks. here the job of the engineer is to have an highers logic overview of what components are needed to be added to the system or improved, that is why langfuse will be used as well for tracing. 

We are trying to use small models and succeed at the task with those one.
Some parts of the code logic will be just like a normal app which doesn't use LLMs but code logic to solve problems or to perform tasks, therefore, sometimes agents will just be the one deciding if that part of the code needs to be activated/used or not.

We are using a Redis cache which will use TTL to expired cache and will store the queries and the response retrieves in double entries one having the key query store as ahsh for exact match search and the double entry with the key query stored as vector for semantic search. This hybrid search permit to increase the chance to find the cached entry. If not, then, the expensive vector database search is done. The fallback is an internet search if the data is not found. 

There is not extraction refining process implemented but would be amazing...

The user query with a sentence or a question, with or without PDF file path or URL in it's query. The agents are going to detect and reformulate the query and process the file or URL. **Only one file or URL can be put per query**. Then if there is a PDF or a URL, a dataframe will be created, then saved to a `.parquet` file. Then the `.parquet` file will be used by another agent to store the data in the database using another chunking process. Now that the database has all the data in with the structure that we want, the embedding is going to start using another chunking process and creating custom documents. Then another agent will get the reformulated query to check if there is something in the cache to start building an answer, if not it will try to retrieve in the vectordb. If it found an answer at about the similarity score level, it will cache it using the hash of the query and another way using the vector of the query so tha it can be perfomed seamntic search and hash matching for next cache retrievals. After all that the report will be created, but before an extra internet search agent will bring some more information, whiule the last agent will use the pydantic structured output in order to have the report written to a file in the format that we want, grouping results as we want it to be. Graph stops, user opens the file which will be located in the Report folder


Docs: [LangGraph Doc Workflow Example](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag_local/)

# Separation of graphs, prompts, utils functions, structured output classes...etc
Everything has been separated in order to make the code closer to what you want to have in production so that you can scale it easily. Graphs can run independently using the `if __name__ == "__main__"`. And the `app.py` file will be used to coordinate everything. It could be a `streamlit` or a `mesop` app.

# installation

- check the `e_n_v_example` and fill it: It is composed by two different env files, one static and one dynamic
- install `Redis`
- install `postgresql` and make sure you have created the database, the user permission and activated `PGVector`
- install `ollama` and get the `mistral:7B` model and make sure that the default model is changed in the library if you see that error set it manually to `mistral:7b` instead of default `llama2`
- install docker as the project as an optional implementation of in docker sandbox code execution that be used to create a tool for agents to execute code safely.
- **Optional:** If using `Langfuse` install it easilly in Docker
- Install dependencies: `pip install -r requirements.txt`
- Start the app: `python3 app.py`

Put your query in:
- if there is an URL in the query, the agents will detect it and start their job
- if there is a pdf path, the agents will detect it and start their job
- **Only one URL or PDF information can be put in the query. The project didn't implement the management of several files or urls. But you can do it by adding some loops probably**
- if there aren't any URL or PDF file in the query, agents will understand and just answer after having searched online a bit.

# Redis Cache Clearance Cron Job
- When a user query the llm, and a retrieval is performed, if the llm can find relevant an answer and use it to answer the query, the content will be cached.
- We will index the frequently retrieved data in a Redis cache and provide the llm the choice to directly answer to user if the cache has the answer
- There were a script `clear_cache.py` will clear Redis cach and also set the `retrieved` colomn from the postgresql table to False.
```bash
chmod +x setup_cron.sh
# will start the script: `clear_cache.py`
./setup_cron.sh
```
**But if TTL is set in Redis no need to clear the cache as Redis is gonna get rid of it according to the TTL.**
- **We set TTL in this project**

### Tips:
- For the learning curve, take the habit to print the states after every `nodes` in order to understand what has been updated in the state and check langfuse traces after implementation


# To do's
- [x] Get boilerplate done and study the different objects by printing them all
- [x] Make a drawing to start in order to get our project plan
- [x] Read more the documentation in order to understand different objects
- [x] Do more of the tutorial to have a feel of different way to use it and print the output to familiarize with the different objects
- [x] Create a function that runs agent code in docker and does it on the fly would be perfect for a tool and safety of kind of sandboxing of agent code
- [x] Change the pdf/webpage_parser docs in order for those to integrate new db rows and structure and store the right content and have the pandas dataframes with right columns
- [x] create nodes and edges
- [x] adapt the function that analyzes query from the beginning to have that step finally done
- [x] Create and test PDF and Webpage parser to get only the quality data and structure it in a good way to enhance embedding/retrieval performances
- [x] Test again all first step of query analysis and parsing of documents and add the step that saves to db uring the right structure
- [x] Test retrieval workflow with caching workflow
- [x] create all the extra functions matching each nodes and test those individually
- [x] test functions along the way in the workflow of agents and make sure of the output type matches and that it works
- [x] create prompts along the way and store those in the prompt file using naming `<name_of_function>_prompt`
- [x] Read comment in `query_matching.py` to create those logics and functions and nodes and tools functions
- [x] create all mini tools
- [x] finish the user query graph until data is embedded or user get response if nothing to embed, therefore, introduce the caching layer
- [x] Find interesting storytelling for use case an interesting use case
- [x] Test last graph `report creation` and `retrieval` workflow by commenting out the rest and putting entry dummy values to see how the report comes out, we can create keys of hash of query to test retrieval also and returned report.
- [x] draw for the video the diagram easy to understand
- [ ] check the `next` just before this one
- [ ] add langfuse
- [ ] make the nodes and conditional edges after code execution conditional edge
- [ ] make all structured outputs needed
- [ ] put all hard coded prompts to the prompt file and import those



