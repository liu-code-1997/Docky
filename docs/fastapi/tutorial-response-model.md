# Response Model

You can declare the model used for the **response** of a path operation. This
controls what data is sent back, validates it, and documents it.

## Declare a response_model

Pass `response_model` to the path operation decorator:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


@app.post("/items/", response_model=Item)
async def create_item(item: Item) -> Item:
    return item
```

FastAPI will take the returned object, validate it against `Item`, and use the
model to generate the response and its OpenAPI schema.

## Filtering output data

A common use is returning a different (usually smaller) model than the input,
so sensitive fields are never sent back. For example, accept a password on
input but never include it in the output:

```python
class UserIn(BaseModel):
    username: str
    password: str
    email: str


class UserOut(BaseModel):
    username: str
    email: str


@app.post("/user/", response_model=UserOut)
async def create_user(user: UserIn):
    return user  # password is stripped from the response by UserOut
```

Even though the function returns the full `UserIn`, the response only contains
the fields declared in `UserOut`.

## response_model_exclude_unset

To omit fields that were not explicitly set (and just have defaults), use
`response_model_exclude_unset=True`. This keeps responses lean, returning only
the values that were actually provided.

```python
@app.get("/items/{item_id}", response_model=Item, response_model_exclude_unset=True)
async def read_item(item_id: str):
    return items[item_id]
```

## Status codes

Set the default HTTP status code for a response with `status_code`:

```python
from fastapi import status


@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(name: str):
    return {"name": name}
```

## Recap

- `response_model` controls, validates, and documents the response shape.
- Use a separate output model to filter out sensitive fields.
- `response_model_exclude_unset` trims default values.
- `status_code` sets the response status.
