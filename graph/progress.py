"""进度广播模块：各处理节点通过此模块向前端推送实时日志。"""

import asyncio

_queues: dict[str, asyncio.Queue] = {}


def register_queue(output_dir: str, queue: asyncio.Queue):
    _queues[output_dir] = queue


def unregister_queue(output_dir: str):
    _queues.pop(output_dir, None)


async def send_log(output_dir: str, message: str):
    """向前端推送一条日志（async 上下文）。"""
    queue = _queues.get(output_dir)
    if queue:
        await queue.put(("log", {"message": message}))


async def send_event(output_dir: str, event_type: str, data: dict):
    """向前端推送一个事件（async 上下文）。"""
    queue = _queues.get(output_dir)
    if queue:
        await queue.put((event_type, data))
