from typing import List, Type

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


def get_all_subclasses(cls):
    all_subclasses = set()

    for subclass in cls.__subclasses__():
        all_subclasses.add(subclass)
        all_subclasses.update(get_all_subclasses(subclass))

    return all_subclasses


def get_all_ops_tools(tool_class: type = DuoBaseTool) -> List[Type[DuoBaseTool]]:
    tool_classes = []

    for subclass in get_all_subclasses(tool_class):
        # Check if the subclass has _execute method
        if hasattr(subclass, "_execute"):
            method = getattr(subclass, "_execute")

            # Check if it's not abstract
            if not getattr(method, "__isabstractmethod__", False):
                tool_classes.append(subclass)

    return tool_classes
