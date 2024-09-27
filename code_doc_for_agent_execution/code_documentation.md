
        '''
        This file contains the documentation in how to write a mardown Python script to make API to satisfy user request.
        Make sure that the script is having the right imports, indentations, uses Python standard libraries and do not have any error before starting writing it.
        '''


      
# Making an API Call to Get the Age of a Friend using Agify API

## Introduction

In this documentation, we will guide you through the process of making an API call to get the age of a friend based on their name using the Agify API.

## Prerequisites

* Python 3.x installed on your system
* `requests` library installed (you can install it using `pip install requests`)

## Step 1: Import the `requests` Library

In your Python script, import the `requests` library using the following code:
```python
import requests
```
## Step 2: Set the API Endpoint and Parameters

Set the API endpoint and parameters using the following code:
```python
api_endpoint = 'https://api.agify.io'
name = 'Junko'
params = {'name': name}
```
## Step 3: Make the API Call

Make a GET request to the API endpoint using the `requests.get()` method and pass the parameters:
```python
response = requests.get(api_endpoint, params=params)
```
## Step 4: Parse the Response

Parse the response using the `response.json()` method:
```python
data = response.json()
```
## Step 5: Extract the Age

Extract the age from the response data:
```python
age = data['age']
```
## Step 6: Print the Age

Print the age of your friend:
```python
print(f'The age of {name} is {age}')
```
## Conclusion

By following these steps, you should be able to make an API call to get the age of your friend based on their name using the Agify API.

