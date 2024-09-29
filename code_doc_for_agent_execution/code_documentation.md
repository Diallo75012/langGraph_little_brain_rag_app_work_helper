
        '''
        This file contains the documentation in how to write a mardown Python script to make API to satisfy user request.
        Make sure that the script is having the right imports, indentations, uses Python standard libraries and do not have any error before starting writing it.
        '''


      
# Getting Started with Agify API

This documentation provides a step-by-step guide on how to make an API call to the Agify API to retrieve the age of a person based on their name.

## Step 1: Import Required Libraries

To make an API call, you need to import the `requests` library, which allows you to send HTTP requests using Python.

```python
import requests
```

## Step 2: Define the API URL

The Agify API requires a name parameter to be passed in the URL. Define the API URL with the name parameter.

```python
api_url = 'https://api.agify.io?name={name}'
```

## Step 3: Define the `get_age_by_name` Function

Create a function that takes a name as input, constructs the API URL with the name parameter, sends a GET request to the Agify API, and parses the JSON response to extract the age.

```python
def get_age_by_name(name):
    # Construct the API URL with the name parameter
    url = api_url.format(name=name)
    
    # Send a GET request to the Agify API
    response = requests.get(url)
    
    # Parse the JSON response
    data = response.json()
    
    # Extract the age from the response
    age = data['age']
    
    return age
```

## Step 4: Make the API Call

Call the `get_age_by_name` function with the name 'Junko' to retrieve the age.

```python
name = 'Junko'
age = get_age_by_name(name)
print(f'The age of {name} is {age}')
```

## Example Output

The output should be the age of the person based on their name.

```
The age of Junko is 25
```

Note: Make sure to handle errors and exceptions properly when making API calls.
