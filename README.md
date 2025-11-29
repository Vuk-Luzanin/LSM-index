# LSM Index System — Database Systems 2 (2024/2025)

## School of Electrical Engineering, University of Belgrade  
Project for the course **Database Systems 2**  
Academic year **2024/2025**

---

## 📌 Project Description

This project implements a system for working with **LSM (Log-Structured Merge Tree)** indexes.  
The system creates and manages an LSM index for any selected column (Di) of the input table (*FactTable*).  
Each indexed column has its own independent LSM index.

The base table contains up to **13,000 rows**, following a schema such as:

```
FactTable(ID, D1, …, Dn, Fact1, …, Factm)
```

- **Di** columns represent dimension attributes and are used for indexing.
- **Facti** columns represent fact (measure) attributes and support aggregations.

---

## 🌲 LSM Index Structure

The LSM tree is organized into multiple levels:

- Level **0** (memtable) holds up to **1000 rows**.
- Each next level **i+1** has **3×** the capacity of level **i**.
- The system supports:
  - **Insert**
  - **Delete**
  - **Equality search**
  - **Merging levels**

---

## 🔍 Query Support

Queries may specify:

- Conditions on one or more indexed columns (Di)
- Logical operators: **AND**, **OR**
- Aggregate functions over non-indexed Fact columns:
  - `MIN`
  - `MAX`
  - `AVG`
  - `SUM`
  - `COUNT`

Queries can be executed:

- **With index usage**
- **Without index usage** (full table scan)

---

## 📥 Input Files

The project reads:

- A table file (CSV)
- Query definitions (text-based input)
- Schema description

In this template project, sample data is provided in:

```
sample_data.csv
```

---

## 📦 Project Structure

```
├── lsm_system.py      # Implementation of the LSM tree and related structures
├── main.py            # Entry point: parsing input, executing queries, showing results
└── sample_data.csv    # Example dataset used for testing
```
