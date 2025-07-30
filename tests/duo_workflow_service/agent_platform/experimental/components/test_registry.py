from unittest.mock import patch

import pytest

from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
)
from duo_workflow_service.agent_platform.experimental.components.registry import (
    ComponentRegistry,
    register_component,
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
