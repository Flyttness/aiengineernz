import inspect
from typing import Callable, TypeVar

from langchain_core.callbacks.base import BaseCallbackHandler
from streamlit.external.langchain.streamlit_callback_handler import StreamlitCallbackHandler
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx


class CustomStreamlitCallbackHandler(StreamlitCallbackHandler):
    def write_agent_name(self, name: str):
        self._parent_container.write(name)


def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """
    Creates a Streamlit callback handler that integrates fully with any LangChain ChatLLM integration,
    updating the provided Streamlit container with outputs such as tokens, model responses,
    and intermediate steps. This function ensures that all callback methods run within
    the Streamlit execution context, fixing the NoSessionContext() error commonly encountered
    in Streamlit callbacks.

    Args:
        parent_container (DeltaGenerator): The Streamlit container where the text will be rendered
                                           during the LLM interaction.
    Returns:
        BaseCallbackHandler: An instance of StreamlitCallbackHandler configured for full integration
                             with ChatLLM, enabling dynamic updates in the Streamlit app.
    """

    # Define a type variable for generic type hinting in the decorator, ensuring the original
    # function and wrapped function maintain the same return type.
    fn_return_type = TypeVar('fn_return_type')

    def add_streamlit_context(fn: Callable[..., fn_return_type]) -> Callable[..., fn_return_type]:
        """
        Decorator to ensure that the decorated function runs within the Streamlit execution context.
        This is necessary for interacting with Streamlit components from within callback functions
        and prevents the NoSessionContext() error by adding the correct session context.

        Args:
            fn (Callable[..., fn_return_type]): The function to be decorated, typically a callback method.
        Returns:
            Callable[..., fn_return_type]: The decorated function that includes the Streamlit context setup.
        """
        # Retrieve the current Streamlit script execution context.
        # This context holds session information necessary for Streamlit operations.
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> fn_return_type:
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)

        return wrapper

    st_cb = CustomStreamlitCallbackHandler(parent_container=parent_container)

    for method_name, method_func in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        # Identify callback methods that respond to LLM events
        if method_name.startswith('on_'):
            setattr(st_cb, method_name, add_streamlit_context(method_func))

    return st_cb
