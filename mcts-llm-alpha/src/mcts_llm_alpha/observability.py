"""Optional Langfuse run/session hierarchy, without credentials in trace payloads."""
import logging
import os
import uuid
from contextlib import contextmanager, ExitStack

log = logging.getLogger(__name__)


class NullObservation:
    def update(self, **fields):
        pass


class SafeObservation:
    def __init__(self, inner):
        self.inner = inner

    def update(self, **fields):
        try:
            self.inner.update(**fields)
        except Exception:
            log.warning('Langfuse update failed', exc_info=False)


class SearchTracer:
    def __init__(self, mode='auto', session_id=None):
        self.session_id = session_id or 'mcts-' + uuid.uuid4().hex
        self.client = None
        self.last_trace_url = None
        configured = all(os.getenv(k) for k in ('LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY'))
        if mode == 'off' or (mode == 'auto' and not configured):
            return
        if not configured:
            raise ValueError('Langfuse enabled: configure LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY')
        if not os.getenv('LANGFUSE_BASE_URL') and os.getenv('LANGFUSE_HOST'):
            os.environ['LANGFUSE_BASE_URL'] = os.environ['LANGFUSE_HOST']
        try:
            from langfuse import get_client
        except ImportError:
            raise RuntimeError('Install langfuse==5.0.4 with Python >=3.11 for tracing') from None
        self.client = get_client()

    @contextmanager
    def observe(self, name, as_type='span', input=None, **fields):
        if self.client is None:
            yield NullObservation()
            return
        from langfuse import propagate_attributes
        stack, observation = ExitStack(), NullObservation()
        try:
            stack.enter_context(propagate_attributes(session_id=self.session_id, tags=['mcts-llm-alpha']))
            observation = stack.enter_context(self.client.start_as_current_observation(
                name=name, as_type=as_type, input=input, **fields))
            self.last_trace_url = self.client.get_trace_url()
        except Exception:
            log.warning('Langfuse observation unavailable', exc_info=False)
        try:
            yield SafeObservation(observation)
        except BaseException as error:
            SafeObservation(observation).update(level='ERROR', status_message=type(error).__name__)
            raise
        finally:
            try:
                stack.close()
            except Exception:
                log.warning('Langfuse observation close failed', exc_info=False)

    def flush(self):
        if self.client:
            try:
                self.client.flush()
            except Exception:
                log.warning('Langfuse flush failed', exc_info=False)
