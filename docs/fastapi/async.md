# Concurrency and async / await

FastAPI supports asynchronous code using Python's `async` and `await`. This
lets your application handle many requests concurrently without blocking.

## async def path operations

You can declare a path operation function with `async def`:

```python
@app.get("/")
async def read_results():
    results = await some_async_library()
    return results
```

Use `async def` when you call libraries that you `await` (async database
drivers, async HTTP clients, etc.).

## Plain def path operations

If your code uses a library that is **not** async (a normal blocking database
driver, for example), just declare the function with plain `def`:

```python
@app.get("/")
def read_results():
    results = some_blocking_library()
    return results
```

FastAPI will run plain `def` functions in an external threadpool so they don't
block the event loop. You don't have to do anything special — both styles work.

## When to use which

- Use `async def` if you need to `await` something inside the function.
- Use plain `def` if you're calling blocking code and can't `await` it.
- If you're not sure, plain `def` is a safe default; FastAPI handles the
  threadpool for you.

## Don't block the event loop

Inside an `async def` function, never call slow blocking code directly (like
`time.sleep()` or a synchronous network call) — it will freeze the event loop
and stall other requests. Either use an async equivalent (`await asyncio.sleep()`)
or move the work into a plain `def` function.

## Recap

- `async def` for awaitable code, plain `def` for blocking code.
- FastAPI runs plain `def` in a threadpool automatically.
- Never call blocking code directly inside `async def`.
