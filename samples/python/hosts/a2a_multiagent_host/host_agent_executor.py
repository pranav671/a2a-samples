import asyncio
import json
import logging

from typing import TYPE_CHECKING

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCard,
    DataPart,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from google.adk import Runner
from google.genai import types


if TYPE_CHECKING:
    from google.adk.sessions.session import Session


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Constants
DEFAULT_USER_ID = 'self'


class HostAgentExecutor(AgentExecutor):
    """An AgentExecutor that runs an ADK-based Agent for host_agent."""

    def __init__(self, runner: Runner, card: AgentCard):
        self.runner = runner
        self._card = card
        self.output = None
        # Track active sessions for potential cancellation
        self._active_sessions: set[str] = set()

    async def _process_request(
        self,
        new_message: types.Content,
        session_id: str,
        task_updater: TaskUpdater,
        response_trace = None,
    ) -> None:
        self.output = None
        session_obj = await self._upsert_session(session_id)
        # Update session_id with the ID from the resolved session object.
        # (it may be the same as the one passed in if it already exists)
        session_id = session_obj.id

        # Track this session as active
        self._active_sessions.add(session_id)

        break_async_loop = False
        try:
            async for event in self.runner.run_async(
                user_id=DEFAULT_USER_ID,
                session_id=session_id,
                new_message=new_message,
            ):
                if break_async_loop:
                    break
                logger.debug(
                    '### Event received: %s',
                    event.model_dump_json(exclude_none=True, indent=2),
                )

                if event.is_final_response():
                    parts = [
                        convert_genai_part_to_a2a(part)
                        for part in event.content.parts
                        if (part.text or part.file_data or part.inline_data)
                    ]

                    logger.debug('#### Yielding final response: %s', parts)
                    await task_updater.add_artifact(parts)

                    await task_updater.update_status(
                        TaskState.completed, final=True
                    )

                    break
                if event.get_function_responses():
                    for response_obj in event.get_function_responses():
                        response = response_obj.model_dump()
                        print("RESPONSE IN HOST AGENT EXECUTOR", response)
                        if response.get('response', {}).get('result', {}).get('status', {}).get('state', '') == 'working':
                            while self.output is None:
                                await asyncio.sleep(5)
                            await task_updater.update_status(
                                TaskState.completed, 
                                message=task_updater.new_agent_message([DataPart(data=self.output)]),
                                final=True,
                            )
                            break_async_loop = True
                        self.output = None

                if not event.get_function_calls():
                    parts = [
                        convert_genai_part_to_a2a(part)
                        for part in event.content.parts
                        if (part.text or part.file_data or part.inline_data)
                    ]

                    logger.debug('#### Yielding update response: %s', parts)
                    await task_updater.update_status(
                        TaskState.working,
                        message=task_updater.new_agent_message(
                            [
                                convert_genai_part_to_a2a(part)
                                for part in event.content.parts
                                if (
                                    part.text
                                    or part.file_data
                                    or part.inline_data
                                )
                            ],
                        ),
                    )
                else:
                    logger.debug('#### Event - Function Calls')
                    agent_name = ''
                    agent_query = ''
                    calls = event.get_function_calls()
                    if calls:
                        for call in calls:
                            if call.name == 'send_message':
                                agent_name = call.args.get('agent_name')
                                agent_query = call.args.get('task')

        finally:
            # Remove from active sessions when done
            self._active_sessions.discard(session_id)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
        response_trace = None,
    ):
        logger.debug('[host_agent] execute called with context: %s', context)
        logger.debug(
            '[host_agent] execute called with context.requested_extensions: %s',
            context.requested_extensions,
        )

        # Run the agent until either complete or the task is suspended.
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        # Immediately notify that the task is submitted.
        if not context.current_task:
            await updater.update_status(TaskState.submitted)
        await updater.update_status(TaskState.working)
        await self._process_request(
            types.UserContent(
                parts=[
                    convert_a2a_part_to_genai(part)
                    for part in context.message.parts
                ],
            ),
            context.context_id,
            updater,
            # traceability=traceability,
            response_trace=response_trace,
        )
        logger.debug('[host_agent] execute exiting')

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Cancel the execution for the given context.

        Currently logs the cancellation attempt as the underlying ADK runner
        doesn't support direct cancellation of ongoing tasks.
        """
        session_id = context.context_id
        if session_id in self._active_sessions:
            logger.info(
                f'Cancellation requested for active host_agent session: {session_id}'
            )
            # TODO: Implement proper cancellation when ADK supports it
            self._active_sessions.discard(session_id)
        else:
            logger.debug(
                f'Cancellation requested for inactive host_agent session: {session_id}'
            )

        raise ServerError(error=UnsupportedOperationError())

    async def _upsert_session(self, session_id: str) -> 'Session':
        """Retrieves a session if it exists, otherwise creates a new one.

        Ensures that async session service methods are properly awaited.
        """
        session = await self.runner.session_service.get_session(
            app_name=self.runner.app_name,
            user_id=DEFAULT_USER_ID,
            session_id=session_id,
        )
        if session is None:
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name,
                user_id=DEFAULT_USER_ID,
                session_id=session_id,
            )
        return session


def convert_a2a_part_to_genai(part: Part) -> types.Part:
    """Convert a single A2A Part type into a Google Gen AI Part type.

    Args:
        part: The A2A Part to convert

    Returns:
        The equivalent Google Gen AI Part

    Raises:
        ValueError: If the part type is not supported
    """
    part = part.root
    if isinstance(part, TextPart):
        return types.Part(text=part.text)
    if isinstance(part, FilePart):
        if isinstance(part.file, FileWithUri):
            return types.Part(
                file_data=types.FileData(
                    file_uri=part.file.uri, mime_type=part.file.mime_type
                )
            )
        if isinstance(part.file, FileWithBytes):
            return types.Part(
                inline_data=types.Blob(
                    data=part.file.bytes, mime_type=part.file.mime_type
                )
            )
        raise ValueError(f'Unsupported file type: {type(part.file)}')
    raise ValueError(f'Unsupported part type: {type(part)}')


def convert_genai_part_to_a2a(part: types.Part) -> Part:
    """Convert a single Google Gen AI Part type into an A2A Part type.

    Args:
        part: The Google Gen AI Part to convert

    Returns:
        The equivalent A2A Part

    Raises:
        ValueError: If the part type is not supported
    """
    if part.text:
        return TextPart(text=part.text)
    if part.file_data:
        return FilePart(
            file=FileWithUri(
                uri=part.file_data.file_uri,
                mime_type=part.file_data.mime_type,
            )
        )
    if part.inline_data:
        return Part(
            root=FilePart(
                file=FileWithBytes(
                    bytes=part.inline_data.data,
                    mime_type=part.inline_data.mime_type,
                )
            )
        )
    raise ValueError(f'Unsupported part type: {part}')
