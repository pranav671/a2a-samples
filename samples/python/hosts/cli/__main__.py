import asyncio
import base64
import os
import urllib
import datetime
from uuid import uuid4

import asyncclick as click
import httpx
from rich import print
from rich.console import Console
from rich.panel import Panel

from a2a.client import A2ACardResolver, A2AClient
from a2a.extensions.common import HTTP_EXTENSION_HEADER
from a2a.types import (
    FilePart,
    FileWithBytes,
    GetTaskRequest,
    JSONRPCErrorResponse,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskQueryParams,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)

console = Console()


def format_stream_event(event: dict) -> str:
    kind = event.get("kind")
    task_id = event.get("taskId", "N/A")
    final = event.get("final", False)

    if kind == "status-update":
        state = event.get("status", {}).get("state", "unknown")
        timestamp = event.get("status", {}).get("timestamp", "unknown")
        timestamp_obj = datetime.datetime.fromisoformat(timestamp)
        timestamp_str = timestamp_obj.strftime("%Y-%m-%d %H:%M:%S")
        status_str = f"[{timestamp_str}] - Status: {state.upper()}"
        console.print(status_str)
        if final:
            if event.get('status', {}).get('message'):
                print(event.get('status', {}).get('message', {}).get('parts', [])[0].get('data', {}))

    elif kind == "artifact-update":
        artifact = event.get("artifact", {})
        parts = artifact.get("parts", [])
        texts = [p.get("text") for p in parts if p.get("kind") == "text" and p.get("text")]
        text_str = " | ".join(texts) if texts else "No text provided"
        console.print(f"Artifact update")
        console.print(f"[bold yellow]{text_str}[/bold yellow]", markup=True)

    else:
        console.print(f"Unknown event type '{kind}' for Task {task_id}")

@click.command()
@click.option('--agent', default='http://localhost:8083/a2a')
@click.option(
    '--bearer-token',
    help='Bearer token for authentication.',
    envvar='A2A_CLI_BEARER_TOKEN',
)
@click.option('--session', default=0)
@click.option('--history', default=False)
@click.option('--use_push_notifications', default=False)
@click.option('--push_notification_receiver', default='http://localhost:5000')
@click.option('--header', multiple=True)
@click.option(
    '--enabled_extensions',
    default='',
    help='Comma-separated list of extension URIs to enable (sets X-A2A-Extensions header).',
)
async def cli(
    agent,
    bearer_token,
    session,
    history,
    use_push_notifications: bool,
    push_notification_receiver: str,
    header,
    enabled_extensions,
):
    headers = {h.split('=')[0]: h.split('=')[1] for h in header}
    if bearer_token:
        headers['Authorization'] = f'Bearer {bearer_token}'

    # --- Add enabled_extensions support ---
    # If the user provided a comma-separated list of extensions,
    # we set the X-A2A-Extensions header.
    # This allows the server to know which extensions are activated.
    # Note: We assume the extensions are supported by the server.
    # This headers will be used by the server to activate the extensions.
    # If the server does not support the extensions, it will ignore them.
    if enabled_extensions:
        ext_list = [
            ext.strip() for ext in enabled_extensions.split(',') if ext.strip()
        ]
        if ext_list:
            headers[HTTP_EXTENSION_HEADER] = ', '.join(ext_list)
    console.print(f'[bold cyan]Using headers:[/] {headers}')
    async with httpx.AsyncClient(timeout=30, headers=headers) as httpx_client:
        card_resolver = A2ACardResolver(httpx_client, agent)
        card = await card_resolver.get_agent_card()

        console.print(Panel(
            'Agent Card',
            style='bold blue',
            border_style='blue',
            expand=False
        ))
        console.print_json(card.model_dump_json(exclude_none=True))

        notif_receiver_parsed = urllib.parse.urlparse(
            push_notification_receiver
        )
        notification_receiver_host = notif_receiver_parsed.hostname
        notification_receiver_port = notif_receiver_parsed.port

        if use_push_notifications:
            from hosts.cli.push_notification_listener import (
                PushNotificationListener,
            )

            push_notification_listener = PushNotificationListener(
                host=notification_receiver_host,
                port=notification_receiver_port,
            )
            push_notification_listener.start()

        client = A2AClient(httpx_client, agent_card=card)

        continue_loop = True
        streaming = card.capabilities.streaming
        context_id = session if session > 0 else uuid4().hex

        while continue_loop:
            console.print('\n' + '='*20 + ' Starting a new task ' + '='*20, style='bold green')
            continue_loop, _, task_id = await completeTask(
                client,
                streaming,
                use_push_notifications,
                notification_receiver_host,
                notification_receiver_port,
                None,
                context_id,
            )

            if history and continue_loop:
                console.print('\n' + '='*20 + ' Task History ' + '='*20, style='bold yellow')
                task_response = await client.get_task(
                    {'id': task_id, 'historyLength': 10}
                )
                console.print_json(
                    task_response.model_dump_json(
                        include={'result': {'history': True}}
                    )
                )


async def completeTask(
    client: A2AClient,
    streaming,
    use_push_notifications: bool,
    notification_receiver_host: str,
    notification_receiver_port: int,
    task_id,
    context_id,
):
    console.print('\n[bold]Enter your message for the agent[/bold]', style='blue')
    console.print('[dim](Type :q or quit to exit)[/dim]')
    prompt = click.prompt('> ', default='', show_default=False)
    if prompt == ':q' or prompt == 'quit':
        return False, None, None

    message = Message(
        role='user',
        parts=[TextPart(text=prompt)],
        message_id=str(uuid4()),
        task_id=task_id,
        context_id=context_id,
    )

    payload = MessageSendParams(
        id=str(uuid4()),
        message=message,
        configuration=MessageSendConfiguration(
            accepted_output_modes=['text'],
        ),
    )

    if use_push_notifications:
        payload['pushNotification'] = {
            'url': f'http://{notification_receiver_host}:{notification_receiver_port}/notify',
            'authentication': {
                'schemes': ['bearer'],
            },
        }

    taskResult = None
    message = None
    task_completed = False
    if streaming:
        response_stream = client.send_message_streaming(
            SendStreamingMessageRequest(
                id=str(uuid4()),
                params=payload,
            )
        )
        async for result in response_stream:
            if isinstance(result.root, JSONRPCErrorResponse):
                print(
                    f'Error: {result.root.error}, context_id: {context_id}, task_id: {task_id}'
                )
                return False, context_id, task_id
            event = result.root.result
            context_id = event.context_id
            if isinstance(event, Task):
                task_id = event.id
            elif isinstance(event, TaskStatusUpdateEvent) or isinstance(
                event, TaskArtifactUpdateEvent
            ):
                task_id = event.task_id
                if (
                    isinstance(event, TaskStatusUpdateEvent)
                    and event.status.state == 'completed'
                ):
                    task_completed = True
            elif isinstance(event, Message):
                message = event
            format_stream_event(event.model_dump())
        # Upon completion of the stream. Retrieve the full task if one was made.
        if task_id and not task_completed:
            taskResultResponse = await client.get_task(
                GetTaskRequest(
                    id=str(uuid4()),
                    params=TaskQueryParams(id=task_id),
                )
            )
            if isinstance(taskResultResponse.root, JSONRPCErrorResponse):
                print(
                    f'Error: {taskResultResponse.root.error}, context_id: {context_id}, task_id: {task_id}'
                )
                return False, context_id, task_id
            taskResult = taskResultResponse.root.result
    else:
        try:
            # For non-streaming, assume the response is a task or message.
            event = await client.send_message(
                SendMessageRequest(
                    id=str(uuid4()),
                    params=payload,
                )
            )
            event = event.root.result
        except Exception as e:
            print('Failed to complete the call', e)
        if not context_id:
            context_id = event.context_id
        if isinstance(event, Task):
            if not task_id:
                task_id = event.id
            taskResult = event
        elif isinstance(event, Message):
            message = event

    if message:
        print(f'\n{message.model_dump_json(exclude_none=True)}')
        return True, context_id, task_id
    if taskResult:
        # Don't print the contents of a file.
        task_content = taskResult.model_dump_json(
            exclude={
                'history': {
                    '__all__': {
                        'parts': {
                            '__all__': {'file'},
                        },
                    },
                },
            },
            exclude_none=True,
        )
        print(f'\n{task_content}')
        ## if the result is that more input is required, loop again.
        state = TaskState(taskResult.status.state)
        if state.name == TaskState.input_required.name:
            return (
                await completeTask(
                    client,
                    streaming,
                    use_push_notifications,
                    notification_receiver_host,
                    notification_receiver_port,
                    task_id,
                    context_id,
                ),
                context_id,
                task_id,
            )
        # task is in-progress 
        if state.name == TaskState.working.name:
            return True, context_id, task_id
        ## task is complete
        return True, context_id, task_id
    ## Failure case, shouldn't reach
    return True, context_id, task_id


if __name__ == '__main__':
    asyncio.run(cli())
