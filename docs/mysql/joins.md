# Joins

A **join** combines rows from two or more tables based on a related column.
Joins are how relational databases connect data that is split across tables.

## Example tables

```sql
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,          -- references users.id
    amount DECIMAL(10, 2)
);
```

Here `orders.user_id` is a **foreign key** pointing at `users.id`.

## INNER JOIN

Returns only rows that have a match in both tables:

```sql
SELECT users.name, orders.amount
FROM users
INNER JOIN orders ON orders.user_id = users.id;
```

A user with no orders, or an order with no matching user, is excluded.

## LEFT JOIN

Returns all rows from the left table, plus matches from the right (or `NULL`
where there's no match):

```sql
SELECT users.name, orders.amount
FROM users
LEFT JOIN orders ON orders.user_id = users.id;
```

This includes users who have placed no orders — their `amount` will be `NULL`.
Use a LEFT JOIN when you want "all X, with their Y if any".

## RIGHT JOIN

The mirror of LEFT JOIN: all rows from the right table, matches from the left.
Less common — you can usually rewrite it as a LEFT JOIN by swapping the tables.

## Joining and aggregating

Joins combine naturally with `GROUP BY`:

```sql
SELECT users.name, COUNT(orders.id) AS order_count
FROM users
LEFT JOIN orders ON orders.user_id = users.id
GROUP BY users.id;
```

This counts each user's orders, including zero for users with none.

## Recap

- `INNER JOIN` keeps only matching rows in both tables.
- `LEFT JOIN` keeps all left rows, filling `NULL` where no match.
- Join on the foreign key = primary key relationship.
- Combine joins with `GROUP BY` to aggregate across tables.
