'''This file contains the documentation in how to write a mardown Python script to make API to satisfy user request.Make sure that the script is having the right imports, indentations, uses Python standard libraries and do not have any error before starting writing it.'''


# Getting Started with Agify API
This documentation provides a step-by-step guide on how to write a Python script to call the Agify API to get the age of a friend based on their name.

## Step 1: Install the `requests` Library
The `requests` library is a popular Python library used to send HTTP requests. You can install it using pip:
```
pip install requests
```
## Step 2: Import the `requests` Library
In your Python script, import the `requests` library:
```
import requests
```
## Step 3: Define the API Endpoint and Parameters
The Agify API endpoint is `https://api.agify.io?name=[name]`. Replace `[name]` with the name of your friend, for example, `Junko`.
```
api_endpoint = 'https://api.agify.io?name=Junko'
```
## Step 4: Send a GET Request to the API Endpoint
Use the `requests.get()` method to send a GET request to the API endpoint:
```
response = requests.get(api_endpoint)
```
## Step 5: Check the Response Status Code
Check if the response was successful (200 OK):
```
if response.status_code == 200:
    # Process the response data
else:
    print('Error:', response.status_code)
```
## Step 6: Parse the Response Data
The Agify API returns a JSON response. Use the `response.json()` method to parse the response data:
```
data = response.json()
```
## Step 7: Extract the Age from the Response Data
Extract the age from the response data:
```
age = data['age']
print('The age of Junko is:', age)
```
That's it! You have now successfully written a Python script to call the Agify API to get the age of your friend based on their name.
