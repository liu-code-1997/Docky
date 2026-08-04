# Dependencies

FastAPI has a powerful **Dependency Injection** system. It lets you declare
the things your path operation functions need, and FastAPI provides them
automatically.

## What is a dependency

A dependency is just a function (or callable) that returns something. You
declare that a path operation "depends on" it, and FastAPI runs it for you
before running your own function, then passes the result in.

## Create a dependency

```python
from fastapi import Depends, FastAPI

app = FastAPI()


async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}


@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

`Depends(common_parameters)` tells FastAPI: before calling `read_items`, call
`common_parameters`, and inject its return value as the `commons` argument.

## Why use dependencies

- **Shared logic**: pagination params, auth checks, DB sessions — declare once,
  reuse across many path operations.
- **Less repetition**: no copy-pasting the same query parameters everywhere.
- **Automatic docs**: parameters declared in a dependency still show up in the
  OpenAPI schema and interactive docs.

## Classes as dependencies

Any callable works, so a class is a common choice — its `__init__` parameters
become request parameters:

```python
class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit


@app.get("/items/")
async def read_items(commons: CommonQueryParams = Depends(CommonQueryParams)):
    return {"q": commons.q, "skip": commons.skip}
```

## Sub-dependencies

Dependencies can themselves depend on other dependencies. FastAPI resolves the
whole tree for you and caches each dependency's result within a single request
(so the same dependency is not computed twice per request).

```python
def query_extractor(q: str | None = None):
    return q


def query_or_default(q: str = Depends(query_extractor)):
    return q or "default"


@app.get("/items/")
async def read_items(final_q: str = Depends(query_or_default)):
    return {"q": final_q}
```

## Dependencies with yield

Use `yield` to run cleanup code after the response is sent — ideal for opening
and closing a database session:

```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/users/")
def read_users(db=Depends(get_db)):
    return db.query(User).all()
```

## Recap

- Declare needs with `Depends(...)`.
- Dependencies can be functions or classes, and can nest.
- Results are cached per request; `yield` dependencies run teardown afterwards.
