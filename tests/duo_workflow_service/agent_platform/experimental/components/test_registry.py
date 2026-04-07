from unittest.mock import Mock, patch

import pytest

from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
)
from duo_workflow_service.agent_platform.experimental.components.registry import (
    ComponentRegistry,
    register_component,
    register_component_factory,
)


class MockBaseComponent(BaseComponent):
    """Mock implementation of BaseComponent for testing."""

    def attach(self, graph, router=None): ...
    def __entry_hook__(self): ...


class TestComponentRegistry:
    """Test suite for ComponentRegistry class."""

    def test_singleton_pattern(self):
        """Test that ComponentRegistry follows singleton pattern."""
        with patch.object(ComponentRegistry, "_instance", None):
            assert ComponentRegistry._instance is None

            registry1 = ComponentRegistry()
            registry2 = ComponentRegistry()
            registry3 = ComponentRegistry.instance()

            assert registry1 is registry2
            assert registry1 is registry3

    def test_non_singleton_pattern(self):
        registry1 = ComponentRegistry(force_new=True)
        registry2 = ComponentRegistry(force_new=True)

        assert registry1 is not registry2

    def test_register_and_get_component_success(self):
        """Test successful component registration."""
        registry = ComponentRegistry(force_new=True)

        registry.register(MockBaseComponent, decorators=[])
        component_class = registry.get("MockBaseComponent")

        assert component_class is MockBaseComponent

    def test_register_component_already_exists_raises_error(self):
        """Test that registering existing component raises ValueError."""
        registry = ComponentRegistry(force_new=True)

        # Register component first time
        registry.register(MockBaseComponent, decorators=[])

        # Try to register again
        with pytest.raises(
            KeyError, match="Component 'MockBaseComponent' is already registered"
        ):
            registry.register(MockBaseComponent, decorators=[])

    def test_get_component_not_found_raises_error(self):
        """Test that getting non-existent component raises KeyError."""
        registry = ComponentRegistry(force_new=True)

        with pytest.raises(
            KeyError, match="Component 'NonExistentComponent' not found in registry"
        ):
            _ = registry["NonExistentComponent"]

    def test_list_registered_components(self):
        """Test listing all registered components."""
        registry = ComponentRegistry(force_new=True)

        # Initially empty
        assert len(registry) == 0

        # Add components
        class Component1(MockBaseComponent):
            pass

        class Component2(MockBaseComponent):
            pass

        registry.register(Component1, decorators=[])
        registry.register(Component2, decorators=[])

        assert len(registry) == 2
        assert "Component1" in registry
        assert "Component2" in registry

    def test_register_with_decorator(self):
        """Test with the decorator."""
        registry = ComponentRegistry(force_new=True)

        class TestComponent(MockBaseComponent):
            wrapped: bool = False

        def decorator(v: type[TestComponent]) -> type[TestComponent]:
            v.wrapped = True
            return v

        registered_type = registry.register(TestComponent, decorators=[decorator])

        assert registered_type.wrapped is True
        assert registry.get("TestComponent") is TestComponent

    def test_register_invalid_class_type(self):
        """Test with non-class object raises TypeError."""
        registry = ComponentRegistry(force_new=True)

        def not_a_class():
            pass

        with pytest.raises(TypeError, match="Invalid component class 'not_a_class'"):
            registry.register(not_a_class, decorators=[])

    def test_register_component_not_basecomponent_subclass(self):
        """Test decorator with class not inheriting from BaseComponent."""
        registry = ComponentRegistry(force_new=True)

        class _NotAComponent:
            pass

        with pytest.raises(TypeError, match="Invalid component class '_NotAComponent'"):
            registry.register(_NotAComponent, decorators=[])


class TestRegisterComponentFactory:
    """Test suite for ComponentRegistry.register_factory method."""

    def test_register_factory_stores_wrapped_callable(self):
        """Factory is stored in the registry under the given name."""
        registry = ComponentRegistry(force_new=True)
        mock_component = Mock(spec=BaseComponent)

        def my_factory(**_kwargs):
            return mock_component

        wrapped = registry.register_factory("MyFactory", my_factory)

        assert "MyFactory" in registry
        assert callable(wrapped)

    def test_register_factory_raises_on_duplicate_name(self):
        """Registering a factory under an already-used name raises KeyError."""
        registry = ComponentRegistry(force_new=True)
        mock_component = Mock(spec=BaseComponent)

        def my_factory(**_kwargs):
            return mock_component

        registry.register_factory("DuplicateFactory", my_factory)

        with pytest.raises(
            KeyError, match="Component 'DuplicateFactory' is already registered"
        ):
            registry.register_factory("DuplicateFactory", my_factory)

    def test_register_factory_validates_return_type(self):
        """Wrapped factory raises TypeError when the underlying factory returns a non-BaseComponent."""
        registry = ComponentRegistry(force_new=True)

        def bad_factory(**_kwargs):
            return "not a component"

        wrapped = registry.register_factory(
            "BadFactory", bad_factory
        )  # bad_factory returns str intentionally to test runtime validation

        with pytest.raises(
            TypeError, match="Factory 'BadFactory' must return a BaseComponent instance"
        ):
            wrapped()

    def test_register_factory_passes_kwargs_to_underlying_factory(self):
        """Wrapped factory forwards all keyword arguments to the underlying factory."""
        registry = ComponentRegistry(force_new=True)
        received_kwargs: dict = {}
        mock_component = Mock(spec=BaseComponent)

        def capturing_factory(**kwargs):
            received_kwargs.update(kwargs)
            return mock_component

        wrapped = registry.register_factory("CapturingFactory", capturing_factory)

        wrapped(foo="bar", baz=42)
        assert received_kwargs == {"foo": "bar", "baz": 42}


class TestRegisterComponentFactoryDecorator:
    """Test suite for register_component_factory standalone decorator."""

    def test_register_component_factory_stores_factory(self):
        """Decorated factory is stored in the registry under the given name."""
        registry = ComponentRegistry(force_new=True)
        mock_component = Mock(spec=BaseComponent)

        with patch.object(ComponentRegistry, "_instance", registry):

            @register_component_factory("MyFactoryDecorated")
            def my_factory(**_kwargs):
                return mock_component

            assert "MyFactoryDecorated" in registry
            assert callable(my_factory)

    def test_register_component_factory_returns_original_callable(self):
        """Decorator returns the original factory callable unchanged."""
        registry = ComponentRegistry(force_new=True)
        mock_component = Mock(spec=BaseComponent)

        with patch.object(ComponentRegistry, "_instance", registry):

            def original_factory(**_kwargs):
                return mock_component

            decorated = register_component_factory("ReturnedFactory")(original_factory)

            # The decorator returns the original function, not the wrapped one
            assert decorated is original_factory

    def test_register_component_factory_raises_on_duplicate_name(self):
        """Registering a factory under an already-used name raises KeyError."""
        registry = ComponentRegistry(force_new=True)
        mock_component = Mock(spec=BaseComponent)

        with patch.object(ComponentRegistry, "_instance", registry):

            @register_component_factory("DuplicateDecoratedFactory")
            def my_factory(**_kwargs):
                return mock_component

            with pytest.raises(
                KeyError,
                match="Component 'DuplicateDecoratedFactory' is already registered",
            ):

                @register_component_factory("DuplicateDecoratedFactory")
                def another_factory(**_kwargs):
                    return mock_component

    def test_register_component_factory_validates_return_type_at_call_time(self):
        """Factory registered via decorator raises TypeError for non-BaseComponent returns."""
        registry = ComponentRegistry(force_new=True)

        with patch.object(ComponentRegistry, "_instance", registry):

            @register_component_factory("BadDecoratedFactory")
            def bad_factory(**_kwargs):
                return "not a component"

            registered = registry["BadDecoratedFactory"]
            with pytest.raises(
                TypeError,
                match="Factory 'BadDecoratedFactory' must return a BaseComponent instance",
            ):
                registered()


class TestRegisterComponentDecorator:
    """Test suite for register_component decorator."""

    def test_register_component(self, component_registry_instance_type):
        """Test decorator."""

        @register_component()
        class TestComponent(MockBaseComponent):
            pass

        component_registry_instance_type.assert_called_once()

        registry = ComponentRegistry.instance()

        # pylint: disable-next=unsupported-membership-test
        assert "TestComponent" in registry
        assert registry.get("TestComponent") is TestComponent

    def test_register_component_with_decorators(self, component_registry_instance_type):
        """Test with additional decorators."""

        def decorator(v: type["TestComponent"]) -> type["TestComponent"]:
            v.wrapped = True
            return v

        @register_component(decorators=[decorator])
        class TestComponent(MockBaseComponent):
            wrapped: bool = False

        component_registry_instance_type.assert_called_once()

        registry = ComponentRegistry.instance()
        registered_class = registry.get("TestComponent")

        # pylint: disable-next=unsupported-membership-test
        assert "TestComponent" in registry
        assert registered_class is TestComponent
        assert registered_class.wrapped is True
