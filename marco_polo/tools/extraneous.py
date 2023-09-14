# Copyright (c) 2023 Mobius Logic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Various uncategorized code"""
import multiprocessing
import sys
from collections.abc import Iterable, Mapping
from multiprocessing.pool import AsyncResult
from typing import Any, Optional


def get_size(obj: Any, seen: Optional[set[Any]] = None) -> int:
    """
    Recursively finds size of objects

    This function has some issues with reference freeing.
    Just be careful calling several times, idk what's going on.

    """
    # https://goshippo.com/blog/measure-real-size-any-python-object/
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


class DummyAsyncResult(AsyncResult[Any]):
    """Dummy 'async' result that is always ready

    This is intended to match the interface so that a non-async item
    can seemlessly replace an async one. This would be used to allow
    a single process execution to use the same code as a multiprocess
    execution.

    Example usage:
    handle = DummyAsyncResult(None, None, None)
    handle.set_result(function())  # function is the 'async' function
    ...
    result = handle.get()  # always ready
    """

    def __init__(self, pool: Any, callback: Any, error_callback: Any) -> None:
        self._pool = pool
        self._event = None
        self._job = None
        self._cache = None
        self._callback = callback
        self._error_callback = error_callback
        self._result = None

    def set_result(self, data: Any) -> None:
        """Set the output of the 'async' run"""
        self._result = data

    def ready(self) -> bool:
        """Return True"""
        return True

    def successful(self) -> bool:
        """Return True"""
        return True

    def wait(self, timeout: Any = None) -> Any:  # pylint: disable=unused-argument
        """Return the result"""
        return self._result

    def get(self, timeout: Any = None) -> Any:  # pylint: disable=unused-argument
        """Return the result"""
        return self._result

    def _set(self, i: Any, obj: Any) -> None:
        pass


class SingleProcessPool(multiprocessing.pool.Pool):  # pylint: disable=abstract-method
    """This is a fake Pool with only a single worker.

    The async jobs are done directly by the main process
    This exists for testing purposes
    """

    def __init__(self) -> None:  # pylint: disable=super-init-not-called
        # this allows the object to be correctly deleted
        self._state = multiprocessing.pool.CLOSE

    def apply_async(  # pylint: disable=too-many-arguments,dangerous-default-value
        self,
        func: Any,
        args: Iterable[Any] = (),
        kwds: Mapping[str, Any] = {},
        callback: Any = None,
        error_callback: Any = None,
    ) -> DummyAsyncResult:
        handle = DummyAsyncResult(None, None, None)
        result = func(*args, **kwds)
        handle.set_result(result)
        return handle
