# pylint: disable=direct-environment-variable-reference

import os

import googlecloudprofiler
import structlog

from duo_workflow_service.tracking.errors import log_exception

logger = structlog.stdlib.get_logger("profiling")


def _setup_pyroscope():
    """Set up Pyroscope profiling for load testing and continuous profiling."""
    try:
        import pyroscope  # pylint: disable=import-outside-toplevel

        server_url = os.environ.get("PYROSCOPE_SERVER_URL")
        if not server_url:
            logger.error("PYROSCOPE_ENABLED=true but PYROSCOPE_SERVER_URL not set")
            return

        application_name = os.environ.get(
            "PYROSCOPE_APPLICATION_NAME", "duo-workflow-service"
        )
        revision = os.environ.get("K_REVISION", "local")

        pyroscope.configure(
            application_name=application_name,
            server_address=server_url,
            tags={
                "revision": revision,
                "region": os.environ.get("K_SERVICE", "unknown"),
            },
            gil_only=False,  # Profile native code too
            enable_logging=True,
        )
        logger.info(
            "Pyroscope profiling enabled",
            server_url=server_url,
            application_name=application_name,
            revision=revision,
        )
    except ImportError:
        logger.error(
            "pyroscope-io not installed. Install with: poetry add pyroscope-io"
        )
    except Exception as e:
        logger.error("Failed to initialize Pyroscope", error=str(e))
        log_exception(e)


def setup_profiling():
    # Google Cloud Profiler (production)
    if os.environ.get("DUO_WORKFLOW_GOOGLE_CLOUD_PROFILER__ENABLED") == "true":
        try:
            googlecloudprofiler.start(
                service="duo-workflow-service",
                service_version=os.environ.get("K_REVISION", "1.0.0"),
            )
            logger.info("Google Cloud Profiler enabled")
        except (ValueError, NotImplementedError) as e:
            log_exception(e)

    # Pyroscope (load testing / continuous profiling)
    if os.environ.get("PYROSCOPE_ENABLED") == "true":
        _setup_pyroscope()
