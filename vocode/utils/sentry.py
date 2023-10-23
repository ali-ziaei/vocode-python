"""TODO move to separate repo."""

import inspect
import os
from functools import wraps
from typing import Dict, Optional, Tuple

import sentry_sdk
from sentry_sdk import set_tag


def sentry_probe(
    name: str,
    description: Optional[str] = None,
    data: Optional[dict] = None,
    tag: Optional[Tuple[str, str]] = None,
):
    """Sentry probe to measure performance of different services Any non-iterable
    function required monitoring, just need to wrap with `sentry_probe` decorator.

    Args:
        name: Name assigned to probe
        description: description of the type or category of operation the span is
            measuring. Default to None
        data: optional data we want to pass to attach to span
        tag: Adding tag to the transaction. It's a tuple of tag key and tag value.
    """
    if tag:
        set_tag(tag[0], tag[1])
    data = data or {}

    def start_child_decorator(func):
        # Asynchronous case
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def func_with_tracing(*args, **kwargs):
                # doing nothing!
                if not os.environ.get("SENTRY_DSN_SECRET_NAME", None):
                    return await func(*args, **kwargs)
                transaction = sentry_sdk.Hub.current.scope.transaction
                if not transaction:
                    with sentry_sdk.start_transaction(name=name, op=description):
                        return await func(*args, **kwargs)

                parent_span = sentry_sdk.Hub.current.scope.span
                if not parent_span:
                    with transaction.start_child(
                        op=name, description=description
                    ) as parent_span:
                        for key, value in data.items():
                            parent_span.set_data(key, value)
                        return await func(*args, **kwargs)

                with parent_span.start_child(
                    op=name, description=description
                ) as child_span:
                    for key, value in data.items():
                        child_span.set_data(key, value)
                    return await func(*args, **kwargs)

        # Synchronous case
        else:

            @wraps(func)
            def func_with_tracing(*args, **kwargs):
                # doing nothing!
                if not os.environ.get("SENTRY_DSN_SECRET_NAME", None):
                    return func(*args, **kwargs)
                transaction = sentry_sdk.Hub.current.scope.transaction
                if not transaction:
                    with sentry_sdk.start_transaction(op=description, name=name):
                        return func(*args, **kwargs)

                parent_span = sentry_sdk.Hub.current.scope.span
                if not parent_span:
                    with transaction.start_child(
                        op=name, description=description
                    ) as parent_span:
                        for key, value in data.items():
                            parent_span.set_data(key, value)
                        return func(*args, **kwargs)

                with parent_span.start_child(
                    op=name, description=description
                ) as child_span:
                    for key, value in data.items():
                        child_span.set_data(key, value)
                    return func(*args, **kwargs)

        return func_with_tracing

    return start_child_decorator


def sentry_probe_async_child_iter(
    name: str,
    description: Optional[str] = None,
    data: Optional[dict] = None,
    tag: Optional[Tuple[str, str]] = None,
):
    """Sentry probe to measure generator performance of different services. The usage is
    the same as "sentry_probe", except this decorator is used only for async generator.
    This function is needed because return with value is not allowed in async generator.

    Args:
        name: Name assigned to prob
        description: description of the type or category of operation the span is
            measuring. Default to None
        data: optional data we want to pass to attach to span
        tag: Adding tag to the transaction. It's a tuple of tag key and tag value.
        trace_each_iter: Whether to add span to each iteration. Default to False
    """
    if tag:
        set_tag(tag[0], tag[1])
    data = data or {}

    async def _iter_func(parent_span, func, *args, **kwargs):
        """Adding span to each iterrable."""
        async for value in func(*args, **kwargs):
            with parent_span.start_child(
                op=f"{name}", description=description
            ) as child_span:
                for key, value in data.items():
                    child_span.set_data(key, value)
                yield value
        return

    def start_child_decorator(func):
        @wraps(func)
        async def async_func_with_tracing(*args, **kwargs):
            if not os.environ.get("SENTRY_DSN_SECRET_NAME", None):
                async for value in func(*args, **kwargs):
                    yield value
                return
            transaction = sentry_sdk.Hub.current.scope.transaction
            if not transaction:
                with sentry_sdk.start_transaction(name=name, op=description) as span:
                    async for value in _iter_func(span, func, *args, **kwargs):
                        yield value
                    return
            cur_span = sentry_sdk.Hub.current.scope.span
            if not cur_span:
                async for value in _iter_func(transaction, func, *args, **kwargs):
                    yield value
                return

            async for value in _iter_func(cur_span, func, *args, **kwargs):
                yield value
            return

        return async_func_with_tracing

    return start_child_decorator


def set_span_data(data: Dict):
    """Set data to current span if span exists."""
    span = sentry_sdk.Hub.current.scope.span
    if span:
        for key, value in data.items():
            span.set_data(key, value)
