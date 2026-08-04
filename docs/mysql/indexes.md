# Indexes

An **index** is a data structure that speeds up row lookups on one or more
columns. Without an index, MySQL must scan the whole table; with one, it can
jump straight to matching rows.

## Why indexes matter

Consider `SELECT * FROM users WHERE email = 'x@y.com'`. On a large table
without an index on `email`, MySQL reads every row (a full table scan). With an
index on `email`, it finds the row almost instantly. Indexes are the single
biggest lever for query performance.

## Create an index

```sql
-- on an existing table
CREATE INDEX idx_users_email ON users (email);

-- or at table creation
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255),
    INDEX idx_email (email)
);
```

## Unique indexes

A unique index also enforces that no two rows share the same value:

```sql
CREATE UNIQUE INDEX idx_users_email ON users (email);
```

The primary key is automatically a unique index.

## Composite indexes

An index can cover several columns. Order matters: a composite index on
`(a, b)` helps queries filtering on `a`, or on `a` and `b`, but not on `b`
alone.

```sql
CREATE INDEX idx_name_created ON users (name, created_at);
```

## The cost of indexes

Indexes speed up reads but slow down writes (`INSERT`/`UPDATE`/`DELETE`),
because the index must be updated too, and they use disk space. Index the
columns you actually filter or join on — not every column.

## Inspecting query plans

Use `EXPLAIN` to see whether a query uses an index:

```sql
EXPLAIN SELECT * FROM users WHERE email = 'x@y.com';
```

Look at the `key` column: if it's `NULL`, no index is being used.

## Recap

- Indexes turn full table scans into fast lookups.
- Index columns used in `WHERE`, `JOIN`, and `ORDER BY`.
- Indexes cost write speed and space, so don't over-index.
- Use `EXPLAIN` to verify an index is used.
