# safe-rl-autodrive

Provably safe reinforcement learning applied to autonomous driving.


# safe-rl-autodrive

Provably safe reinforcement learning applied to autonomous driving.

## Project Setup with Docker and Poetry

This project uses Poetry to manage dependencies, some of which are hosted on private GitLab repositories. To build the Docker image successfully, you'll need to set up SSH key forwarding to authenticate with GitLab during the build process.

### Setup Instructions

#### Add Your SSH Key to the SSH Agent

    First, ensure your SSH key is available for forwarding during the Docker build.

1. **Start the SSH agent**:

    ```bash
    eval "$(ssh-agent -s)"
    ```

2. **Add your SSH private key to the agent:**

    ```bash
    ssh-add ~/.ssh/id_rsa
    ```

3. **Verify the key is added:**

    ```bash
    ssh-add -l
    ```

    You should see your SSH key listed.


#### Build the Docker Image with SSH Forwarding

Use **Docker BuildKit** to forward your SSH agent during the build process.

1. **Enable BuildKit and build the base image:**

    ```bash
     DOCKER_BUILDKIT=1 docker build --ssh default -f Dockerfile.base -t autodrive-base .
    ```

2. **Build the runner image:**

    ```bash
     DOCKER_BUILDKIT=1 docker build --ssh default -f Dockerfile.runtime -t autodrive-run .
    ```

    Building without cache will force poetry to update the dependencies

    ```bash
    DOCKER_BUILDKIT=1 docker build --ssh default -f Dockerfile.runtime --no-cache -t autodrive-run .
    ```


#### Run the Docker Container

Once the image is built, you can run the container as usual:

```bash
docker run --rm -it autodrive
```

Or running it with certain volumes:

```bash
docker run -v ./logs:/autodrive/logs -v ./scripts:/autodrive/scripts -v ./scenarios:/autodrive/scenarios -it autodrive
```

Run a python script unsing poetry:

```bash
poetry run python path/to/file.py
```
e.g

```bash
poetry run python scripts/run_baseline.py --run_name commonroad_ppo_1 --n_epochs 50 --logger_type tensorboard
```

Or to start a poetry shell:

```bash
source $(poetry env info --path)/bin/activate
```


#### User Docker Context

Deploying the Docker Container on a remote server can be simplified by using Docker Contexts. This enables building, running and interacting with Docker Containers which are deployed on the server directly from the host machine. Be aware that login onto the server using a SSH-Agent has to be configured.

1. **Create a new context:**

    ```bash
    docker context create my-remote-server --docker "host=ssh://user@remote-server"
    ```

2. **Use the context:**

    ```bash
    docker context use my-remote-server
    ```

3. **Run docker commands directly on the server from your host machine:**

    ```bash
    docker ps
    ```

    