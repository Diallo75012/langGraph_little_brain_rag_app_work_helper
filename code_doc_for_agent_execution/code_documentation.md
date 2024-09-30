'''This file contains the documentation in how to write a mardown Python script to make API to satisfy user request.Make sure that the script is having the right imports, indentations, uses Python standard libraries and do not have any error before starting writing it.'''


# Making an API Call to Get the Age of a Friend using Agify API

## Introduction

In this documentation, we will guide you through the process of making an API call to get the age of a friend based on their name using the Agify API.

## Prerequisites

* Python installed on your machine
* `requests` library installed (you can install it by running `pip install requests` in your command prompt or terminal)

## Step 1: Import the `requests` Library

In your Python script, import the `requests` library by adding the following line of code:
```python
import requests
```
## Step 2: Define the API Endpoint and Parameters

The Agify API endpoint is `https://api.agify.io?name=[name]`. We will replace `[name]` with the actual name of our friend, which is 'Junko'.

## Step 3: Make a GET Request

Use the `requests` library to make a GET request to the API endpoint. You can do this by adding the following code:
```python
response = requests.get('https://api.agify.io?name=Junko')
```
## Step 4: Check the Response Status Code

Check if the response was successful by checking the status code. If the status code is 200, it means the request was successful.
```python
if response.status_code == 200:
    # Process the response data
else:
    print('Error:', response.status_code)
```
## Step 5: Parse the Response Data

The response data is in JSON format. You can parse it using the `json()` method provided by the `requests` library.
```python
data = response.json()
```
## Step 6: Extract the Age from the Response Data

The response data contains the age of our friend. You can extract it by accessing the relevant key in the `data` dictionary.
```python
age = data['age']
print('The age of Junko is:', age)
```
## Conclusion

By following these steps, you should be able to make an API call to the Agify API and get the age of your friend based on their name.

