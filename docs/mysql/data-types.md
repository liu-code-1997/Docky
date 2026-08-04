# Data Types

Every column in a MySQL table has a **data type** that defines what kind of
values it can store. Choosing the right type saves space and prevents invalid
data.

## Numeric types

- `INT` — standard integer (about ±2.1 billion). Use `UNSIGNED` to double the
  positive range and forbid negatives.
- `BIGINT` — larger integer, for very large counts or ids.
- `TINYINT` — small integer; `TINYINT(1)` is commonly used as a boolean.
- `DECIMAL(p, s)` — exact fixed-point number, ideal for money (e.g.
  `DECIMAL(10, 2)`). Avoid `FLOAT`/`DOUBLE` for currency because they are
  approximate.

## String types

- `VARCHAR(n)` — variable-length string up to `n` characters. Good default for
  names, emails, titles.
- `CHAR(n)` — fixed-length string, padded to `n`. Use only for values that are
  always the same length (e.g. country codes).
- `TEXT` — large text (articles, descriptions). Cannot have a default value and
  is stored differently from `VARCHAR`.

## Date and time types

- `DATE` — a date (`YYYY-MM-DD`).
- `DATETIME` — date and time, not tied to a timezone.
- `TIMESTAMP` — date and time, stored in UTC and converted to the session
  timezone; often used with `DEFAULT CURRENT_TIMESTAMP`.

## Boolean

MySQL has no true boolean type; `BOOLEAN` is an alias for `TINYINT(1)`, where 0
is false and non-zero is true.

## NULL and defaults

A column allows `NULL` (missing value) unless declared `NOT NULL`. Provide a
`DEFAULT` to fill in a value when none is supplied:

```sql
status VARCHAR(20) NOT NULL DEFAULT 'active'
```

## Recap

- Use `INT`/`BIGINT` for integers, `DECIMAL` for money.
- `VARCHAR` for most strings, `TEXT` for long text.
- `DATETIME`/`TIMESTAMP` for time; `TIMESTAMP` is timezone-aware.
- Use `NOT NULL` and `DEFAULT` to keep data clean.
