# LangGraph

- [x] Get boilerplate done and study the different objects by printing them all
- [x] Make a drawing to start in order to get our project plan
- [x] Read more the documentation in order to understand different objects
- [x] Do more of the tutorial to have a feel of different way to use it and print the output to familiarize with the different objects
- [x] Create a function that runs agent code in docker and does it on the fly would be perfect for a tool and safety of kind of sandboxing of agent code
- [ ] Change the pdf/webpage_parser docs in order for those to integrate new db rows and structure and store the right content and have the pandas dataframes with right columns
- [ ] Create and test PDF and Webpage parser to get only the quality data and structure it in a good way to enhance embedding/retrieval performances
- [ ] From experience gathered, start thinking of a custom graph that you want to create. You should be able to make a simple one and then complicate it along the way
- [ ] Be creative and create your tutorial on an interesting use case
Docs: [LangGraph Doc Workflow Example](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag_local/)


# Redis Cache Clearance Cron Job
- When a user query the llm, and a retrieval is performed, if the llm can find relevant an answer and use it to answer the query, the content will be cached.
- We will index the frequently retrieved data in a Redis cache and provide the llm the choice to directly answer to user if the cache has the answer
- This script `clear_cache.py` will clear Redis cach and also set the `retrieved` colomn from the postgresql table to False.
```bash
chmod +x setup_cron.sh
# will start the script: `clear_cache.py`
./setup_cron.sh
```
But if TTL is set in Redis no need to clear the cache as Redis is gonna get rid of it according to the TTL.

### Tips:
- For the learning curve, take the habit to print the states after every `nodes` in order to understand what has been updated in the state

