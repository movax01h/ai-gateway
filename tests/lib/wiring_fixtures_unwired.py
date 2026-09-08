"""Sample @inject target with a marker no test container provides."""

from dependency_injector.wiring import Provide, inject


@inject
def needs_missing(value=Provide["missing"]):
    return value
