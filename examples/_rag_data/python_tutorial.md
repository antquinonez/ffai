# Python Async Programming Tutorial

## Introduction

Asynchronous programming is a programming paradigm that allows your program to handle multiple tasks concurrently without blocking. In Python, this is primarily achieved through the `asyncio` library and the `async`/`await` syntax introduced in Python 3.5.

This tutorial covers the fundamentals of async programming in Python, from basic coroutines to advanced patterns like async context managers and task groups.

## Basic Coroutines

A coroutine is a function defined with `async def`. Unlike regular functions, coroutines don't run immediately when called. Instead, they return a coroutine object that must be awaited.

```python
import asyncio

async def fetch_data(url: str) -> str:
    print(f"Fetching {url}...")
    await asyncio.sleep(1)  # Simulate network delay
    print(f"Done fetching {url}")
    return f"Data from {url}"

async def main():
    result = await fetch_data("https://example.com")
    print(result)

asyncio.run(main())
```

The `await` keyword pauses the coroutine until the awaited operation completes. During this pause, the event loop can run other coroutines.

## Running Tasks Concurrently

The real power of async programming comes from running multiple operations concurrently. The `asyncio.gather()` function runs multiple awaitables concurrently and returns their results in order.

```python
async def fetch_all():
    results = await asyncio.gather(
        fetch_data("https://api1.example.com"),
        fetch_data("https://api2.example.com"),
        fetch_data("https://api3.example.com"),
    )
    return results
```

This runs all three fetches concurrently, completing in roughly 1 second instead of 3.

## Error Handling

When a coroutine raises an exception, it propagates to the caller just like synchronous code.

```python
async def risky_operation():
    await asyncio.sleep(0.5)
    raise ValueError("Something went wrong")

async def safe_main():
    try:
        result = await risky_operation()
    except ValueError as e:
        print(f"Caught error: {e}")
```

## Timeouts

Use `asyncio.wait_for()` to impose a timeout on async operations:

```python
async def with_timeout():
    try:
        result = await asyncio.wait_for(
            fetch_data("https://slow.example.com"),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        print("Operation timed out!")
```

## Task Groups (Python 3.11+)

Task groups provide a structured way to manage concurrent tasks. If any task fails, all remaining tasks are cancelled.

```python
async def process_urls(urls: list[str]):
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch_data(url)) for url in urls]
    results = [task.result() for task in tasks]
    return results
```

## Async Context Managers

Async context managers use `async with` and are useful for managing resources that need async setup or teardown:

```python
class AsyncDatabaseConnection:
    async def __aenter__(self):
        self.conn = await create_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()

async def query_database():
    async with AsyncDatabaseConnection() as db:
        result = await db.execute("SELECT * FROM users")
        return result
```

## Best Practices

1. Always use `asyncio.run()` as the entry point for async programs.
2. Avoid blocking calls (like `time.sleep()`) inside async functions. Use `asyncio.sleep()` instead.
3. Use `asyncio.gather()` for concurrent I/O operations.
4. Handle exceptions explicitly - unhandled exceptions in tasks can silently fail.
5. Consider using task groups for structured concurrency in Python 3.11+.
6. Profile your async code to ensure it's actually faster than synchronous alternatives.

## Common Pitfalls

- Forgetting to `await` a coroutine (it won't execute)
- Using blocking I/O (like `requests.get()`) instead of async alternatives
- Creating too many concurrent tasks (use semaphores to limit concurrency)
- Not handling exceptions in gathered tasks
