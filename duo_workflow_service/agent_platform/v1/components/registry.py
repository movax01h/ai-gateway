from collections.abc import Mapping
from typing import Callable, Optional, Self, Sequence, TypeAlias

from duo_workflow_service.agent_platform.v1.components.base import BaseComponent

__all__ = ["ComponentRegistry", "register_component"]


ComponentClassAlias: TypeAlias = type[BaseComponent] | Callable[..., BaseComponent]
DecoratorAlias: TypeAlias = Callable[[ComponentClassAlias], ComponentClassAlias]


class ComponentRegistry(Mapping):
    """Singleton registry for managing BaseComponent classes.

    This registry implements the singleton pattern to ensure a single global
    registry for all component classes. Components can be registered and retrieved
    by name, enabling dynamic component loading and management.

    The registry acts as a centralized storage for component classes that can be
    used throughout the application. It supports decorator application during
    registration and provides a mapping interface for easy access.

    Example:
        >>> registry = ComponentRegistry.instance()
        >>> registry.register(MyComponent, [some_decorator])
    """

    _instance: Optional[Self] = None

    def __new__(cls, force_new: bool = False):
        """Create a new instance or return singleton based on usage."""
        if force_new:
            # Return new instance without storing it
            return super().__new__(cls)

        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self, force_new: bool = False):
        """Initialize the registry.

        Sets up the internal registry dictionary. This method ensures that
        each instance (whether singleton or new) has its own registry state.

        Args:
            force_new: If True, always initialize a new registry.
                If False, only initialize if not already initialized.
        """
        # Always initialize for new instances, or if not already initialized for singleton
        if force_new or not hasattr(self, "_registry"):
            self._registry: dict[str, ComponentClassAlias] = {}

    @classmethod
    def instance(cls) -> Self:
        """Get the singleton instance of ComponentRegistry.

        This is the preferred method for accessing the global component registry.
        It ensures that the same registry instance is used throughout the application.

        Returns:
            The singleton ComponentRegistry instance.

        Example:
            >>> registry = ComponentRegistry.instance()
            >>> # All subsequent calls return the same instance
            >>> same_registry = ComponentRegistry.instance()
            >>> assert registry is same_registry
        """
        if cls._instance is None:
            cls._instance = cls(force_new=True)
        return cls._instance

    def register(
        self,
        value: type[BaseComponent],
        decorators: Sequence[DecoratorAlias],
    ) -> ComponentClassAlias:
        """Register a component class with optional decorators.

        Registers a component class in the registry using the class name as the key.
        Applies any provided decorators to the class before registration. This method
        performs validation to ensure the component is properly structured.

        Args:
            value: The component class to register. Must inherit from BaseComponent.
            decorators: Sequence of decorator functions to apply to the component
                before registration. Decorators are applied in order.

        Returns:
            The registered component class (potentially modified by decorators).

        Raises:
            KeyError: If a component with the same name is already registered.
            TypeError: If the component doesn't inherit from BaseComponent.

        Example:
            >>> registry = ComponentRegistry.instance()
            >>> decorated_class = registry.register(MyComponent, [inject_decorator])
            >>> # Component is now available as registry["MyComponent"]
        """
        if not issubclass(value, BaseComponent):
            raise TypeError(
                f"Invalid component class '{value.__name__}'. Components must inherit from BaseComponent class"
            )

        register_name = value.__name__

        for decorator in decorators:
            value: ComponentClassAlias = decorator(value)  # type: ignore[no-redef]

        if register_name in self._registry:
            raise KeyError(
                f"Component '{register_name}' is already registered. Use a different name"
            )

        self._registry[register_name] = value

        return value

    def __getitem__(self, key: str, /) -> ComponentClassAlias:
        """Retrieve a registered component class by name.

        Implements the mapping interface to allow dictionary-style access
        to registered components.

        Args:
            key: The name of the component to retrieve.

        Returns:
            The component class registered under the given name.

        Raises:
            KeyError: If no component is registered under the given name.

        Example:
            >>> registry = ComponentRegistry.instance()
            >>> component_class = registry["MyComponent"]
            >>> instance = component_class(name="my_instance", ...)
        """
        klass = self._registry.get(key, None)
        if not klass:
            raise KeyError(f"Component '{key}' not found in registry")

        return klass

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self):
        yield from self._registry.__iter__()


def register_component(
    decorators: Optional[Sequence[DecoratorAlias]] = None,
) -> Callable:
    """Decorator to automatically register a component class with the ComponentRegistry.

    This decorator provides a convenient way to register component classes with the
    global ComponentRegistry instance. It can optionally apply decorators (such as
    dependency injection decorators) to the component before registration.

    The decorator uses the class name as the registration key, so component class
    names must be unique within the application.

    Args:
        decorators: Optional sequence of decorator functions to apply to the
            component class before registration. Common use cases include
            dependency injection decorators. If None, no decorators are applied.

    Returns:
        A decorator function that registers the component and returns the
        potentially modified class.

    Raises:
        KeyError: If a component with the same name is already registered.
        TypeError: If the decorated object is not a class or doesn't inherit
            from BaseComponent.

    Example:
        Basic registration:
        >>> @register_component()
        ... class MyComponent(BaseComponent):
        ...     pass

        Registration with dependency injection:
        >>> from dependency_injector.wiring import inject
        >>> @register_component(decorators=[inject])
        ... class AnotherComponent(BaseComponent):
        ...     def __init__(self, service: SomeService = Provide[...]):
        ...         super().__init__()
        ...         self.service = service

        Multiple decorators:
        >>> @register_component(decorators=[inject, some_other_decorator])
        ... class ComplexComponent(BaseComponent):
        ...     pass
    """

    def decorator(cls: type[BaseComponent]) -> ComponentClassAlias:
        registry = ComponentRegistry.instance()
        registered_class = registry.register(cls, decorators if decorators else [])

        return registered_class

    return decorator
