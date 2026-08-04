# Transactions

A **transaction** groups several SQL statements into a single all-or-nothing
unit of work. Either every statement succeeds and the changes are saved, or
something fails and everything is rolled back as if nothing happened.

## Why transactions matter

The classic example is a money transfer: subtract from one account, add to
another. If the second statement fails after the first succeeded, money would
vanish. A transaction guarantees both happen or neither does.

```sql
START TRANSACTION;

UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

COMMIT;
```

`COMMIT` makes the changes permanent. If anything goes wrong before it, you
issue `ROLLBACK` to undo them all:

```sql
START TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- something is wrong
ROLLBACK;   -- no change is saved
```

## ACID properties

Transactions give you the ACID guarantees:

- **Atomicity**: all statements happen or none do.
- **Consistency**: the database moves from one valid state to another.
- **Isolation**: concurrent transactions don't see each other's half-finished
  work.
- **Durability**: once committed, changes survive a crash.

## Engine matters

Transactions require a transactional storage engine. MySQL's default engine
**InnoDB** supports them; the older **MyISAM** engine does not. Make sure your
tables use InnoDB if you rely on transactions.

## Isolation levels

MySQL lets you tune how strictly concurrent transactions are isolated
(`READ COMMITTED`, `REPEATABLE READ`, etc.). Stricter isolation prevents more
anomalies but can reduce concurrency. InnoDB's default is `REPEATABLE READ`.

## Recap

- A transaction is all-or-nothing: `START TRANSACTION` … `COMMIT` / `ROLLBACK`.
- They provide the ACID guarantees.
- Use InnoDB (not MyISAM) for transaction support.
