# flake8: noqa
from gitlab_cloud_connector import CloudConnectorConfig

from ai_gateway import api, container, main, models
from ai_gateway.config import *

# Set a default service name
CloudConnectorConfig.set_service_name("gitlab-ai-gateway")
