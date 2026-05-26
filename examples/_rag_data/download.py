"""Download sample documents for RAG examples.

Downloads public-domain text from Project Gutenberg to use as input for
chunking, embedding, and retrieval examples.

Usage:
    python -m examples._rag_data.download
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_DIR = Path(__file__).parent

DOCUMENTS: dict[str, str] = {
    "alice_wonderland.txt": "https://www.gutenberg.org/files/11/11-0.txt",
    "python_tutorial.md": "",
    "api_docs.md": "",
}


def _create_python_tutorial() -> str:
    return """# Python Async Programming Tutorial

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
"""


def _create_api_docs() -> str:
    return """# Authentication API Reference

## Overview

The Authentication API provides endpoints for user registration, login, token management, and session handling. All endpoints accept and return JSON payloads.

Base URL: `https://api.example.com/v2/auth`

## Authentication Methods

### Bearer Token

Include a valid JWT token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

Tokens expire after 3600 seconds (1 hour). Use the refresh endpoint to obtain a new token without re-authenticating.

### API Key

For server-to-server communication, include your API key in the header:

```
X-API-Key: <your-api-key>
```

API keys do not expire but can be revoked from the dashboard.

## Endpoints

### POST /register

Register a new user account.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| email | string | Yes | Valid email address |
| password | string | Yes | Minimum 12 characters, 1 uppercase, 1 number, 1 special character |
| name | string | No | Display name (defaults to email prefix) |

**Response (201 Created):**

```json
{
  "user_id": "usr_abc123",
  "email": "user@example.com",
  "created_at": "2025-01-15T10:30:00Z"
}
```

**Error Responses:**
- `400 Bad Request` - Invalid email format or weak password
- `409 Conflict` - Email already registered

Rate limit: 5 requests per minute per IP address.

### POST /login

Authenticate an existing user and receive access tokens.

**Request Body:**

```json
{
  "email": "user@example.com",
  "password": "your-secure-password"
}
```

**Response (200 OK):**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "rt_def456...",
  "expires_in": 3600,
  "token_type": "Bearer"
}
```

**Error Responses:**
- `401 Unauthorized` - Invalid credentials
- `429 Too Many Requests` - Rate limit exceeded (10 per minute)

### POST /refresh

Exchange a refresh token for a new access token.

**Request Body:**

```json
{
  "refresh_token": "rt_def456..."
}
```

**Response (200 OK):**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_in": 3600,
  "token_type": "Bearer"
}
```

Refresh tokens are valid for 30 days. After that, users must re-authenticate.

### DELETE /session

Invalidate the current session and all associated tokens.

**Headers:** `Authorization: Bearer <token>`

**Response (204 No Content):** No response body on success.

This endpoint revokes both the access token and the associated refresh token. All subsequent requests with these tokens will receive `401 Unauthorized`.

## Rate Limiting

All authentication endpoints enforce rate limiting:

| Endpoint | Limit | Window |
|----------|-------|--------|
| /register | 5 requests | 1 minute |
| /login | 10 requests | 1 minute |
| /refresh | 30 requests | 1 minute |
| /session | 5 requests | 1 minute |

When rate limited, the API returns `429 Too Many Requests` with a `Retry-After` header indicating when to retry.

## Error Format

All errors follow RFC 7807 Problem Details format:

```json
{
  "type": "https://api.example.com/errors/invalid-credentials",
  "title": "Authentication Failed",
  "status": 401,
  "detail": "The email or password provided is incorrect.",
  "timestamp": "2025-01-15T10:30:00Z"
}
```
"""


def download(force: bool = False) -> dict[str, Path]:
    os.makedirs(DATA_DIR, exist_ok=True)
    paths: dict[str, Path] = {}

    for name, url in DOCUMENTS.items():
        dest = DATA_DIR / name
        if dest.exists() and not force:
            paths[name] = dest
            continue

        if name == "python_tutorial.md":
            dest.write_text(_create_python_tutorial(), encoding="utf-8")
        elif name == "api_docs.md":
            dest.write_text(_create_api_docs(), encoding="utf-8")
        elif url:
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, dest)
        else:
            continue

        paths[name] = dest
        print(f"  Saved: {dest} ({dest.stat().st_size:,} bytes)")

    return paths


if __name__ == "__main__":
    paths = download()
    print(f"\n{len(paths)} documents ready in {DATA_DIR}/")
    for name, path in paths.items():
        print(f"  {name}: {path.stat().st_size:,} bytes")
