"""
This script will be used by the agent which needs to execute its generated code.
it is Python language only for the moment.
and code will be executed in Docker for safety reasons
"""
import os
from typing import Tuple, Optional


import subprocess

# function to check if image exist for clean up
def check_if_image_exists(image_name: str) -> bool:
    """Check if a Docker image exists."""
    try:
        # Run the 'docker images' command to list all images
        result = subprocess.run(['docker', 'images', '-q', image_name], capture_output=True, text=True, check=False)
        
        # If there's output, the image exists
        return bool(result.stdout.strip())
    
    except Exception as e:
        print(f"An error occurred while checking for the Docker image: {e}")
        return False

def run_script_in_docker(name_of_dockerfile_to_run_script: str, agent_script_file_path: str, requirements_script_file_path: str = "") -> Tuple[str, str]:
    """
      Will copy agent created Python script to build the Dockerfile with it to execute the script in a docker sandbox for secure code execution.
      Creates Dockerfile, builds it and runs the container `sandbox-python`
      When the code execution is done, it deletes everything and stores code execution result in a Tuple (stdout, stderr)
      
      Parameters:
      name_of_dockerfile_to_run_script str: name of the dockerfile
      agent_script_file_path str: the name of the script created by the agent
      
      Output:
      stdout, stderr Tuple[str, str]:
    """

    # Write the script content to a file if needed otherwise just copy the files in the docker file and make agent write those files in this same folder
    #with open("agent_created_script_to_execute_in_docker.py", "w", encoding="utf-8") as going_to_docker_script_file, open(agent_script_file_path, "r", encoding="utf-8") as script_content:
        #going_to_docker_script_file.write(script_content.read())

    # Create the dockerfile for the agent to run the script inside docker
    with open(name_of_dockerfile_to_run_script, "w", encoding="utf-8") as docker_file:
        docker_file.write("# Docker fly created on the fly to execute code in docker by agents.\n")
        docker_file.write("FROM python:3.9-slim\n")
        if requirements_script_file_path != "":
          docker_file.write(f"COPY {requirements_script_file_path} .\n")
          requirements_file_name = requirements_script_file_path.split("/")[-1].strip()
          docker_file.write(f"RUN pip install --no-cache-dir -r {requirements_file_name}\n")
        llm_script_file_name = agent_script_file_path.split("/")[-1].strip()
        docker_file.write(f"COPY {agent_script_file_path} /app/{llm_script_file_name}\n")
        # if env vars in workflow add this line and make agent creating it
        # docker_file.write("COPY .sandbox.env /app/.sandbox.env\n")
        docker_file.write("WORKDIR /app\n")
        docker_file.write(f'CMD ["python", "{llm_script_file_name}"]')

    try:
        # Build the Docker image
        build_command = ['docker', 'build', '-t', 'sandbox-python', '-f', f'{name_of_dockerfile_to_run_script}', '.']
        subprocess.run(build_command, check=True)

        # Run the Docker container and capture the output and delete the container
        run_command = ['docker', 'run', '--rm', 'sandbox-python']
        result = subprocess.run(run_command, capture_output=True, text=True)

        stdout, stderr = result.stdout, result.stderr

        with open("sandbow_docker_execution_logs.md", "w", encoding="utf-8") as script_execution_result:
          script_execution_result.write("# Python script executed in docker, this is the result of captured stdout and stderr")
          script_execution_result.write("""
            This is the result after the execution of the code
            Returns:
            stdout str: the standard output of the script execution which runs in docker, therefore, we capture the stdout to know if the script output is as expected. You need to analyze it and see why it is empty if empty, why it is not as expected to suggest code fix, and if the script executes correctly, get this stdout value and answer using markdown ```markdown ``` saying just one word: OK'
            stderr str: the standard error of the script execution. If this value is not empty answer with the content of the value with a suggestion in how to fix it. Answer using mardown and put a JSON of the error message with key ERROR all between this ```markdown ```. 
          """)
          script_execution_result.write(f"\n\nstdout: {stdout}\nstderr: {stderr}")
        
        # Return the captured output
        return stdout, stderr

    except subprocess.CalledProcessError as e:
        stdout, stderr = '', str(e)
   
    finally:
        # Safely attempt to remove the Docker image only if it exists
        try:
            if check_if_image_exists("sandbox-python"):
                try:
                    cleanup_command = ['docker', 'rmi', '-f', "sandbox-python"]
                    subprocess.run(cleanup_command, check=False)
                    print("Image sandbox-python removed successfully.")
                except Exception as e:
                    print(f"An error occurred while trying to remove the image: {e}")
            else:
                print(f"Image sandbox-python does not exist, skipping cleanup.")

        except subprocess.CalledProcessError as e:
            print(f"Image is not there so all good! We tried to cleanup if any image .. Error during Docker cleanup: {e}")

'''
if __name__ == "__main__":

  #print(run_script_in_docker("test_dockerfile", "./app.py"))
'''







