"""Sample @inject targets for container-wiring tests (not collected by pytest)."""

from dependency_injector.wiring import Provide, inject


@inject
def needs_present(value=Provide["present"]):
    return value
