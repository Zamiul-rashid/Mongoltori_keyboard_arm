# Setting Up Anaconda Environment

This guide will help you set up an Anaconda environment using the provided configuration file.

## Prerequisites

- Anaconda or Miniconda installed on your system.

## Steps

1. **Download the Configuration File**

    Ensure you have the configuration file (`environment.yml`) ready. If you don't have it, download it from the provided source.

2. **Open Terminal**

    Open your terminal (or Anaconda Prompt if you are on Windows).

3. **Navigate to the Directory**

    Navigate to the directory where your `environment.yml` file is located. Use the `cd` command:

    ```sh
    cd /path/to/your/environment.yml
    ```

4. **Create the Environment**

    Run the following command to create the environment from the `environment.yml` file:
    Make sure the `environment.yml` and `requirements.txt` files are in the same directory as your terminal's current location. If they are in a different directory, provide the relative path to these files in the commands.

    For example, if the files are in a folder named `anaconda_envs` within your current directory, use:

    ```sh
    conda env create -f anaconda_envs/environment.yml
    ```

    And for installing pip dependencies:

    ```sh
    pip install -r anaconda_envs/requirements.txt
    ```

5. **Activate the Environment**

    Once the environment is created, activate it using:

    ```sh
    conda activate your_environment_name
    ```

    Replace `your_environment_name` with the name specified in the `environment.yml` file.
6. **Install Additional Dependencies**

    There also is a `requirements.txt` file with additional pip dependencies, install them using the following command after activating your environment:

        ```sh
        pip install -r requirements.txt
        ```

7. **Verify the Installation**

    To ensure everything is set up correctly, you can list all installed packages:

    ```sh
    conda list
    ```

## Additional Tips

- To deactivate the environment, use:

  ```sh
  conda deactivate
  ```

- To remove the environment, use:

  ```sh
  conda env remove -n your_environment_name
  ```

  Replace `your_environment_name` with the name of your environment.

That's it! Your Anaconda environment should now be set up and ready to use.