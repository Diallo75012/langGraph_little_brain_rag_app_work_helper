import os
from typing import List, Dict, Union, Optional, Callable, Tuple


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
        docker_file.write("COPY .env /app/.env\n")
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


print(run_script_in_docker("test_dockerfile", "./app.py"))
