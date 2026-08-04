# Handling Errors

There are many situations where you need to tell the client that something
went wrong: an item doesn't exist, the user isn't authorized, the input is
invalid. FastAPI uses `HTTPException` for this.

## Raise an HTTPException

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

items = {"foo": "The Foo Wrestlers"}


@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item": items[item_id]}
```

When you `raise HTTPException`, FastAPI stops the request and returns an HTTP
response with that status code and a JSON body like `{"detail": "Item not found"}`.

## Add custom headers

Some responses need extra headers, e.g. for certain security requirements:

```python
raise HTTPException(
    status_code=404,
    detail="Item not found",
    headers={"X-Error": "There goes my error"},
)
```

## Custom exception handlers

You can register a handler for your own exception types with
`@app.exception_handler(...)`:

```python
from fastapi import Request
from fastapi.responses import JSONResponse


class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name


@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something."},
    )


@app.get("/unicorns/{name}")
async def read_unicorn(name: str):
    if name == "yolo":
        raise UnicornException(name=name)
    return {"unicorn_name": name}
```

## Validation errors

When request data fails validation, FastAPI automatically returns a `422
Unprocessable Entity` response describing exactly which fields were wrong. You
usually don't need to handle these yourself, but you can override the handler
for `RequestValidationError` if you want a custom format.

## Recap

- Raise `HTTPException(status_code=..., detail=...)` to return an error.
- Add `headers=` for custom response headers.
- Register `@app.exception_handler(...)` for your own exception types.
- Invalid request data yields an automatic `422` response.
