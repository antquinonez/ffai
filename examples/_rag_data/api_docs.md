# Authentication API Reference

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
