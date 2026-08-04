# MySQL

MySQL is a widely used open-source **relational database management system**
(RDBMS). It stores data in tables made of rows and columns, and you query and
manipulate it using SQL (Structured Query Language).

## What MySQL is for

MySQL is the backbone of countless web applications. It's a good fit when your
data is **structured** and **relational** — records that reference each other,
where you want strong consistency and the ability to query with SQL.

## Core concepts

- **Database**: a named container of tables.
- **Table**: a set of rows, each with the same columns.
- **Row (record)**: one entry in a table.
- **Column (field)**: one attribute, with a fixed data type.
- **Primary key**: a column (or set of columns) that uniquely identifies each row.
- **Index**: a structure that speeds up lookups on certain columns.

## Connect with the client

```bash
mysql -u root -p
```

This opens the interactive SQL shell after prompting for the password.

## Create a database and table

```sql
CREATE DATABASE shop;
USE shop;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

- `AUTO_INCREMENT` makes `id` fill in automatically for each new row.
- `PRIMARY KEY` marks the unique identifier.
- `NOT NULL` requires a value; `UNIQUE` forbids duplicates.

## Recap

- MySQL is a relational database queried with SQL.
- Data lives in tables of typed columns.
- Every table should have a primary key.
