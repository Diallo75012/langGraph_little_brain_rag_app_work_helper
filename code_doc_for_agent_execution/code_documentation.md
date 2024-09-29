'''This file contains the documentation in how to write a mardown Python script to make API to satisfy user request.Make sure that the script is having the right imports, indentations, uses Python standard libraries and do not have any error before starting writing it.'''


# How to Make an API Call to Agify.io to Get the Age of a Friend

## Step 1: Install the `requests` Library

To make API calls in Python, we need to install the `requests` library. If you haven't installed it yet, run the following command in your terminal or command prompt:
```
pip install requests
```
## Step 2: Define the `get_age_by_name` Function

Create a new Python function called `get_age_by_name` that takes a name as input. This function will construct the API URL with the name parameter, send a GET request to Agify.io API, and parse the JSON response to extract the age.

```python
import requests

def get_age_by_name(name):
    # Construct the API URL with the name parameter
    api_url = f'https://api.agify.io?name={name}'
    # Send a GET request to the API
    response = requests.get(api_url)
    # Parse the JSON response
    data = response.json()
    # Extract the age from the response
    age = data['age']
    return age
```
## Step 3: Call the `get_age_by_name` Function

Call the `get_age_by_name` function with the name 'Junko' as input:
```python
age = get_age_by_name('Junko')
print(f'The age of Junko is {age}')
```
## Step 4: Run the Script

Run the script to make the API call and get the age of Junko.

That's it! You should now have a Python script that makes an API call to Agify.io to get the age of a friend based on their name.
