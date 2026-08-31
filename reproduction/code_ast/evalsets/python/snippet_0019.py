import os
import pathlib
import shutil
import time
from urllib.parse import urljoin

import invoke
import requests
import requests.auth

CLI_DIR = os.path.join(".", "cli")
JENKINS_BASE_URL = os.environ["JENKINS_URL"]
JENKINS_HOME_DIR = os.path.join(".", "jenkins_home")
JENKINS_PASSWORD = os.environ["<PASSWORD>"]
JENKINS_USER = os.environ["JENKINS_LEADER_ADMIN_USER"]
PLUGINS_FILE = os.path.join(".", "jenkins_plugins.txt")

JENKINS_AUTH = requests.auth.HTTPBasicAuth(JENKINS_USER, JENKINS_PASSWORD)
JENKINS_CLI_BASE = f"docker-compose exec cli java -jar jenkins-cli.jar -auth {JENKINS_USER}:{JENKINS_PASSWORD}"

with open(PLUGINS_FILE) as pfh:
    REQUIRED_PLUGINS = [l.strip() for l in pfh.readlines()]


@invoke.task
def create(ctx):
    """
    Create the Jenkins environment

    Creates Jenkins server container.

    Creates Jenkins cli container.

    Installs Jenkins plugins and verifies they were installed correctly.

    :param ctx: Context object passed by invoke to a task
    :return: None
    """
    jenkins_master_init(ctx)

    jenkins_cli_init(ctx)

    install_jenkins_plugins(ctx, plugins=REQUIRED_PLUGINS)

    diff = verify_jenkins_plugins_are_installed(plugins=REQUIRED_PLUGINS)
    if diff:
        print(f"All required plugins were not installed. Missing: {diff}")
        exit(24)

    create_seed_job(ctx)


@invoke.task
def destroy(ctx):
    """
    Destroy the Jenkins environment

    Destroys all containers.

    Removes directories created by the create task.

    :param ctx: Context object passed by invoke to a task
    :return: None
    """
    ctx.run("docker-compose down", hide=True)
    shutil.rmtree(CLI_DIR, ignore_errors=True)
    shutil.rmtree(JENKINS_HOME_DIR, ignore_errors=True)


def check_jenkins_available():
    """
    Checks whether REST API to the Jenkins server is available

    :return: Boolean
    """
    url = urljoin(JENKINS_BASE_URL, "/")

    try:
        result = requests.get(url, auth=JENKINS_AUTH)
    except requests.exceptions.ConnectionError:
        return False

    return result.status_code == 200


def jenkins_master_init(ctx):
    """
    Make Jenkins server ready for use

    :param ctx: Context object passed to the parent task by invoke
    :return: None
    """
    os.makedirs(JENKINS_HOME_DIR, exist_ok=True)

    os.makedirs(os.path.join(JENKINS_HOME_DIR, "init.groovy.d"), exist_ok=True)
    shutil.copy2(os.path.join(".", "jenkins_admin_user.groovy"),
                 os.path.join(JENKINS_HOME_DIR, "init.groovy.d", "admin_user.groovy"))
    shutil.copy2(os.path.join(".", "jenkins_csrf.groovy"),
                 os.path.join(JENKINS_HOME_DIR, "init.groovy.d", "csrf.groovy"))
    shutil.copy2(os.path.join(".", "jenkins_harden.groovy"),
                 os.path.join(JENKINS_HOME_DIR, "init.groovy.d", "harden.groovy"))

    jenkins_master_container_up(ctx)


def jenkins_cli_init(ctx):
    """
    Make Jenkins cli ready for use

    :param ctx: Context object passed to the parent task by invoke
    :return: None
    """
    download_jenkins_cli()
    jenkins_cli_container_up(ctx)


def jenkins_master_container_up(ctx):
    """
    Bring up the Jenkins server container

    Since docker-compose is used here, the lifecycle of image and container is offloaded to it.

    :param ctx: Context object passed to the parent task by invoke
    :return: None
    """
    ctx.run("docker-compose up -d jenkins", hide=True)
    # print("RUNNING")

    available = check_jenkins_available()

    while not available:
        available = check_jenkins_available()
        time.sleep(5)

    # print('AVAILABLE')


def install_jenkins_plugins(ctx, plugins=None):
    """
    Install required plugins using Jenkins CLI

    Uses the -deploy flag so Jenkins server does not need to be restarted to make the plugins active.

    More info: https://jenkins.io/doc/book/managing/plugins/#install-with-cli

    An alternative way is to run the following command in the Jenkins container. Its biggest drawback is that, unlike
    Jenkins CLI, it does not "deploy" the plugin(s) after install thus requires a restart to make them active.
    Also requires a workaround identified in https://github.com/jenkinsci/docker/issues/348.

            /usr/local/bin/install-plugins.sh plugin1 plugin2 ...

    :param ctx: Context object passed to the parent task by invoke
    :param plugins: List of plugins to install
    :return: None
    """
    cmd = f"{JENKINS_CLI_BASE} install-plugin"
    for plugin in plugins:
        ctx.run(f"{cmd} {plugin} -deploy", pty=True, hide=True)


def restartjenkins(ctx):
    """
    Safely restart Jenkins server using cli

    :return: None
    """
    print("LOG: Restart Jenkins")
    ctx.run(f"{JENKINS_CLI_BASE} safe-restart",
            pty=True, hide=True)

    def is_up():
        for _ in range(0, 10):
            available = check_jenkins_available()
            print(f"LOG: Available: {available}")
            if available:
                break
            print(f"LOG: Sleep 5s")
            time.sleep(5)
        else:
            print("Jenkins is not up")
            exit(7)

    is_up()

    # The following method uses REST API instead of cli

    # quiet_down_url = urljoin(JENKINS_BASE_URL, "quietDown")
    # quiet_down_result = requests.post(quiet_down_url, auth=JENKINS_AUTH)
    #
    # if quiet_down_result.status_code != 200:
    #     print("Quiet Down API call failed")
    #     exit(12)
    #
    # restart_url = urljoin(JENKINS_BASE_URL, "restart")
    # restart_result = requests.post(restart_url, auth=JENKINS_AUTH)
    #
    # if restart_result.status_code == 503 and "Please wait while Jenkins is restarting" in restart_result.text:
    #     pass
    # elif restart_result.status_code != 200:
    #     print("Restart API call failed")
    #     exit(13)
    #
    # is_up()

    time.sleep(10)  # Extra time for Jenkins to be fully up


def verify_jenkins_plugins_are_installed(plugins):
    """
    Verify that all expected plugins are installed in Jenkins server

    :param plugins: List of plugins expected to be installed
    :return: List of missing plugins
    """
    url = urljoin(JENKINS_BASE_URL, "pluginManager/api/json?depth=1")
    result = requests.get(url, auth=JENKINS_AUTH)

    if result.status_code != 200:
        print("API call to get list of installed plugins failed")
        exit(17)

    installed_plugins = [plugin["shortName"] for plugin in result.json()["plugins"]]

    diff = set(plugins) - set(installed_plugins)

    return sorted(diff)


def download_jenkins_cli():
    """
    Download cli jar from the Jenkins server

    :return: None
    """
    os.makedirs(CLI_DIR, exist_ok=True)
    destination = os.path.join(CLI_DIR, "jenkins-cli.jar")

    if not pathlib.Path(destination).is_file():
        url = urljoin(JENKINS_BASE_URL, "jnlpJars/jenkins-cli.jar")
        download_file(url, dest=destination, auth=JENKINS_AUTH)


def jenkins_cli_container_up(ctx):
    """
    Bring up the Jenkins cli container

    Since docker-compose is used here, the lifecycle of image and container is offloaded to it.

    :param ctx: Context object passed to the parent task by invoke
    :return: None
    """
    ctx.run("docker-compose up -d cli", hide=True)


def download_file(url, dest, auth=None):
    """
    Download a file from a web server

    :param url: Link from where the file is to be downloaded
    :param dest: Path (including file name) where the file is to be created
    :param auth: Optional HTTP Basic Auth object
    :return: None
    """
    result = requests.get(url, auth=auth, stream=True)
    with open(dest, 'wb') as dfh:
        for chunk in result.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive
                dfh.write(chunk)


def create_seed_job(ctx):
    """
    Create seed job on Jenkins server

    :param ctx: Context object passed to the parent task by invoke
    :return: None
    """
    # Use *exec -T* because a file (seed-job.xml) is being read and piped to
    # the docker-compose command.
    # When *exec -T* is not used, docker-compose gives an error
    # "the input device is not a TTY" and the command does not succeed,
    # even when ctx.run() is given pty=True like so: ctx.run(cmd, pty=True).
    cmd = "docker-compose exec -T cli java -jar jenkins-cli.jar " + \
          f"-auth {JENKINS_USER}:{JENKINS_PASSWORD} " + \
          "create-job seed < ./seed-job.xml"
    ctx.run(cmd, hide=True)