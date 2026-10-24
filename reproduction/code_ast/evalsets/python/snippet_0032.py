"""Custom settings module for sansa-desa's data_cube_ui

We use this mostly to fetch additional settings from the environment

"""

import os
import typing
from configparser import ConfigParser
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

from .settings import *


def get_env_variable(
        variable: str,
        default: typing.Optional[typing.Any] = None,
        use_prefix: typing.Optional[bool] = True
):
    prefix = 'DJANGO__' if use_prefix else ''
    env_var_name = f'{prefix}{variable}'
    value = os.getenv(env_var_name)
    if value is None:
        if default:
            value = default
        else:
            raise ImproperlyConfigured(
                f'Add the {env_var_name!r} environment variable')
    return value


def get_bool_env_variable(variable, default: typing.Optional[bool] = None):
    try:
        value = get_env_variable(variable, default=str(default))
    except ImproperlyConfigured:
        raise
    truthy_values = (
        '1',
        'on',
        'yes',
        'true',
    )
    return True if value.lower() in truthy_values else False


def get_odc_db_connection_details(
        config_path: Path,
        env: typing.Optional[str] = 'default'
) -> typing.Dict[str, str]:
    config = ConfigParser()
    config.read(config_path)
    return {
        'host': config[env].get('db_hostname'),
        'port': config[env].get('db_port'),
        'db_name': config[env]['db_database'],
        'username': config[env]['db_username'],
        'password': config[env]['db_password'],
    }


DEBUG = get_bool_env_variable('DEBUG', False)

SECRET_KEY = get_env_variable('SECRET_KEY')

ADMIN_EMAIL = get_env_variable('ADMIN_EMAIL', ADMIN_EMAIL)

TIME_ZONE = 'UTC'

STATIC_ROOT = get_env_variable(
    'STATIC_ROOT', str(Path('~').expanduser() / 'data_cube_ui_static_root'))

STATICFILES_DIRS = [
    str(Path(BASE_DIR) / 'static'),
]

CELERY_BROKER_URL = get_env_variable(
    'CELERY_BROKER_URL', 'redis://localhost:6379/0')

CELERY_RESULT_BACKEND = get_env_variable(
    'CELERY_RESULT_BACKEND', CELERY_BROKER_URL)

DC_UI_DIR = str(Path(BASE_DIR) / 'utils')

DATA_CUBE_UI_RESULTS_DIR = get_env_variable(
    'DATA_CUBE_UI_RESULTS_DIR',
    str(Path(BASE_DIR) / 'ui_results')
)

DATABASES['default'].update({
    'NAME': get_env_variable('DEFAULT_DB_NAME', 'data_cube_ui'),
    'USER': get_env_variable('DEFAULT_DB_USER', DATABASES['default']['NAME']),
    'PASSWORD': get_env_variable(
        'DEFAULT_DB_PASSWORD',
        DATABASES['default']['USER']
    ),
    'HOST': get_env_variable('DEFAULT_DB_HOST', 'localhost'),
    'PORT': get_env_variable('DEFAULT_DB_PORT', 5432),
})

# get ODC DB connection details from the already existing odc configuration file
DATACUBE_CONFIG_PATH = get_env_variable(
    'DATACUBE_CONFIG_PATH',
    default=Path('~/.datacube.conf').expanduser(),
    use_prefix=False
)
DATACUBE_ENVIRONMENT = get_env_variable(
    'DATACUBE_ENVIRONMENT',
    default='internal',
    use_prefix=False
)
odc_db_details = get_odc_db_connection_details(DATACUBE_CONFIG_PATH, DATACUBE_ENVIRONMENT)
DATABASES['agdc'].update({
    'NAME': odc_db_details['db_name'],
    'USER': odc_db_details['username'],
    'PASSWORD': odc_db_details['password'],
    'HOST': odc_db_details['host'],
    'PORT': odc_db_details['port']
})