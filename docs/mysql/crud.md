# CRUD Operations

CRUD stands for Create, Read, Update, Delete — the four basic operations on
data. In MySQL these map to `INSERT`, `SELECT`, `UPDATE`, and `DELETE`.

## Create — INSERT

```sql
INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com');

-- insert several rows at once
INSERT INTO users (name, email) VALUES
    ('Bob', 'bob@example.com'),
    ('Carol', 'carol@example.com');
```

## Read — SELECT

```sql
-- all columns, all rows
SELECT * FROM users;

-- specific columns with a condition
SELECT name, email FROM users WHERE id = 1;

-- filtering, ordering, limiting
SELECT name FROM users
WHERE name LIKE 'A%'
ORDER BY created_at DESC
LIMIT 10;
```

`WHERE` filters rows, `ORDER BY` sorts, and `LIMIT` caps how many rows come
back. `LIKE 'A%'` matches names starting with A.

## Update — UPDATE

```sql
UPDATE users
SET email = 'alice@new.com'
WHERE id = 1;
```

Always include a `WHERE` clause. An `UPDATE` without `WHERE` changes **every
row** in the table.

## Delete — DELETE

```sql
DELETE FROM users WHERE id = 1;
```

As with `UPDATE`, a `DELETE` without `WHERE` removes all rows. Double-check the
condition before running it.

## Counting and aggregating

```sql
SELECT COUNT(*) FROM users;
SELECT status, COUNT(*) FROM users GROUP BY status;
```

`GROUP BY` collapses rows that share a value so you can aggregate per group.

## Recap

- `INSERT` to create, `SELECT` to read, `UPDATE` to modify, `DELETE` to remove.
- Always use `WHERE` with `UPDATE` and `DELETE` to avoid touching every row.
- `ORDER BY`, `LIMIT`, and `GROUP BY` shape query results.
