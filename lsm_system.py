
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import csv
import json
import re
import time

# ----------------------------
# ENUMS I DATACLASS
# ----------------------------

class LogicalOperator(Enum):
    AND = "AND"
    OR = "OR"

class AggregateFunction(Enum):
    MIN = "min"
    MAX = "max"
    AVG = "avg"
    SUM = "sum"
    COUNT = "count"

@dataclass
class SearchCondition:
    column: str
    value: Any

@dataclass
class SearchQuery:
    conditions: List[SearchCondition]
    logical_op: LogicalOperator
    aggregate_column: Optional[str] = None
    aggregate_function: Optional[AggregateFunction] = None
    use_index: bool = True

@dataclass
class TableRow:
    id: int
    dimensions: Dict[str, Any]
    facts: Dict[str, float]
    is_tombstone: bool = False

# ----------------------------
# JEDNOSTAVNI B+ STABLO INDeks
# ----------------------------

class BPlusTree:
    def __init__(self, order: int = 50):
        self.data: Dict[Any, List[int]] = {}

    def insert_secondary_index(self, value: Any, row_id: int):
        if value not in self.data:
            self.data[value] = []
        if row_id not in self.data[value]:
            self.data[value].append(row_id)

    def remove(self, value: Any, row_id: int) -> bool:
        if value in self.data and row_id in self.data[value]:
            self.data[value].remove(row_id)
            if not self.data[value]:
                del self.data[value]
            return True
        return False

    def get_row_ids(self, value: Any) -> List[int]:
        return self.data.get(value, [])

    def keys(self):
        return list(self.data.keys())

# ----------------------------
# LSM INDEKS SA TOMBSTONE
# ----------------------------

class ColumnIndex:
    TOMBSTONE_PREFIX = "#"

    def __init__(self, column_name: str):
        self.column_name = column_name
        self.index = BPlusTree(order=50)

    def insert(self, value: Any, row_id: int):
        self.index.insert_secondary_index(value, row_id)

    def delete(self, value: Any, row_id: int):
        tombstone_value = f"{self.TOMBSTONE_PREFIX}{value}"
        self.index.insert_secondary_index(tombstone_value, row_id)

    def search(self, value: Any) -> List[int]:
        ids = []
        tombstones = set(self.index.get_row_ids(f"{self.TOMBSTONE_PREFIX}{value}"))
        for rid in self.index.get_row_ids(value):
            if rid not in tombstones:
                ids.append(rid)
        return ids

    def keys(self):
        return list(self.index.keys())

# ----------------------------
# LSM NIVO
# ----------------------------

class LSMLevel:
    def __init__(self, level_num: int, max_capacity: int):
        self.level_num = level_num
        self.max_capacity = max_capacity
        self.rows: List[TableRow] = []
        self.is_sorted = True

    def add_row(self, row: TableRow) -> bool:
        self.rows.append(row)
        self.is_sorted = False
        return len(self.rows) > self.max_capacity

    def sort_by_id(self):
        if not self.is_sorted:
            self.rows.sort(key=lambda r: r.id)
            self.is_sorted = True

    def get_all_rows(self) -> List[TableRow]:
        return self.rows.copy()

    def clear(self):
        self.rows.clear()
        self.is_sorted = True

# ----------------------------
# LSM SISTEM
# ----------------------------

class LSMIndexSystem:
    def __init__(self, table_schema: Dict[str, List[str]]):
        self.dimension_columns = table_schema["dimensions"]
        self.fact_columns = table_schema["facts"]
        self.levels: List[LSMLevel] = []
        capacities = [1000, 3000, 9000]
        for i, cap in enumerate(capacities):
            self.levels.append(LSMLevel(i, cap))

        self.index_columns = self.dimension_columns
        self.index_levels = len(self.levels)
        self.column_indexes: Dict[str, List[ColumnIndex]] = {
            col: [ColumnIndex(col) for _ in range(self.index_levels)]
            for col in self.index_columns
        }

        self.all_rows: Dict[int, TableRow] = {}
        self.next_row_id = 1
        self.deleted_rows: Set[int] = set()

    # ------------------------
    # INSERT / SPILL
    # ------------------------

    def insert_row(self, dimensions: Dict[str, Any], facts: Dict[str, float], row_id: Optional[int] = None) -> int:
        if row_id is None:
            row_id = self.next_row_id
            self.next_row_id += 1
        else:
            self.next_row_id = max(self.next_row_id, row_id + 1)

        row = TableRow(row_id, dimensions, facts)
        self.all_rows[row_id] = row
        self._insert_into_lsm(row)
        return row_id

    def _insert_into_lsm(self, row: TableRow):
        overflowed = self.levels[0].add_row(row)
        self._update_index_for_row(0, row)
        if overflowed:
            self._spill_level(0)

    def _spill_level(self, level_num: int):
        if level_num >= len(self.levels) - 1:
            return

        current_level = self.levels[level_num]
        next_level = self.levels[level_num + 1]

        current_level.sort_by_id()
        next_level.sort_by_id()

        tombstone_ids = {row.id for row in current_level.rows if row.is_tombstone}
        tombstone_ids.update(row.id for row in next_level.rows if row.is_tombstone)

        filtered_rows = [row for row in current_level.rows if not row.is_tombstone and row.id not in tombstone_ids]

        space_in_next = next_level.max_capacity - len(next_level.rows)
        if space_in_next <= 0:
            self._spill_level(level_num + 1)
            space_in_next = next_level.max_capacity - len(next_level.rows)

        to_move = min(space_in_next, len(filtered_rows))
        moving_rows = filtered_rows[:to_move]
        next_level.rows.extend(moving_rows)

        current_level.rows = [row for row in current_level.rows if row.is_tombstone or row not in moving_rows]

        self._rebuild_index_for_level(level_num)
        self._rebuild_index_for_level(level_num + 1)

        if len(next_level.rows) > next_level.max_capacity:
            self._spill_level(level_num + 1)

    # ------------------------
    # INDEKSI
    # ------------------------

    def _update_index_for_row(self, level_num: int, row: TableRow):
        for col in self.index_columns:
            idx_obj: ColumnIndex = self.column_indexes[col][level_num]
            if col in row.dimensions:
                if row.is_tombstone:
                    idx_obj.delete(row.dimensions[col], row.id)
                else:
                    idx_obj.insert(row.dimensions[col], row.id)

    def _rebuild_index_for_level(self, level_num: int):
        for col in self.index_columns:
            self.column_indexes[col][level_num] = ColumnIndex(col)
        for row in self.levels[level_num].get_all_rows():
            for col in self.index_columns:
                if col in row.dimensions:
                    if row.is_tombstone:
                        self.column_indexes[col][level_num].delete(row.dimensions[col], row.id)
                    else:
                        self.column_indexes[col][level_num].insert(row.dimensions[col], row.id)

    # ------------------------
    # DELETE
    # ------------------------

    def delete_row(self, row_id: int) -> bool:
        if row_id not in self.all_rows:
            return False
        row = self.all_rows[row_id]
        tombstone_row = TableRow(row_id, row.dimensions.copy(), row.facts.copy(), True)
        self._insert_into_lsm(tombstone_row)
        return True

    # ------------------------
    # SEARCH
    # ------------------------

    def search_with_index(self, query: SearchQuery) -> List[TableRow]:
        if not query.conditions:
            return []

        cond_results = []

        for cond in query.conditions:
            if cond.column not in self.column_indexes:
                return self.search_without_index(query)

            valid_row_ids = set()
            for lvl in range(self.index_levels):
                idx: ColumnIndex = self.column_indexes[cond.column][lvl]
                ids = idx.search(cond.value)
                valid_row_ids.update(ids)
            cond_results.append(valid_row_ids)

        if query.logical_op == LogicalOperator.AND:
            result_ids = set.intersection(*cond_results) if cond_results else set()
        else:
            result_ids = set.union(*cond_results) if cond_results else set()

        tombstone_ids = set()
        for lvl in range(self.index_levels):
            for col in self.index_columns:
                idx: ColumnIndex = self.column_indexes[col][lvl]
                for key in idx.keys():
                    if str(key).startswith(ColumnIndex.TOMBSTONE_PREFIX):
                        ids = idx.index.get_row_ids(key) or []
                        tombstone_ids.update(ids)

        result_ids = [rid for rid in result_ids if rid in self.all_rows and rid not in tombstone_ids]
        return [self.all_rows[rid] for rid in result_ids]

    def search_without_index(self, query: SearchQuery) -> List[TableRow]:
        result = []
        for rid, row in self.all_rows.items():
            if row.is_tombstone:
                continue
            if self._row_matches_conditions(row, query.conditions, query.logical_op):
                result.append(row)
        return result

    def _row_matches_conditions(self, row: TableRow, conditions: List[SearchCondition], logical_op: LogicalOperator) -> bool:
        results = []
        for cond in conditions:
            if cond.column == "Id":
                match = row.id == cond.value
            elif cond.column in row.dimensions:
                match = row.dimensions[cond.column] == cond.value
            elif cond.column in row.facts:
                match = row.facts[cond.column] == cond.value
            else:
                match = False
            results.append(match)
        return all(results) if logical_op == LogicalOperator.AND else any(results)

    # ------------------------
    # SELECT / UPDATE / DELETE sa benchmark
    # ------------------------

    def execute_select(self, columns: List[str], conditions: List[SearchCondition],
                    logical_op: LogicalOperator, use_index=True,
                    aggregate_function: Optional[AggregateFunction] = None,
                    aggregate_column: Optional[str] = None,
                    raw_output: bool = False,
                    benchmark: bool = True) -> Any:

        query = SearchQuery(
            conditions=conditions,
            logical_op=logical_op,
            use_index=use_index,
            aggregate_function=aggregate_function,
            aggregate_column=aggregate_column
        )

        # Odabir pretrage sa ili bez indeksa
        if use_index:
            rows_index = self.search_with_index(query)
            if benchmark:
                start_full = time.time()
                rows_full = self.search_without_index(query)
                time_full = (time.time() - start_full) * 1000
                # Benchmark ispisujemo samo za SELECT, ne za UPDATE/DELETE
                if not aggregate_function and not columns == []:
                    print()#f"[BENCHMARK] Sa indeksom: {(time.time() - start_full)*1000:.2f} ms, Bez indeksa: {time_full:.2f} ms")
            rows = rows_index
        else:
            rows = self.search_without_index(query)

        result = []

        # Obrada agregatnih funkcija
        if aggregate_function and aggregate_column:
            values = [r.facts[aggregate_column] for r in rows if aggregate_column in r.facts]

            if aggregate_function == AggregateFunction.COUNT:
                agg_result = len(values)
            elif values:
                if aggregate_function == AggregateFunction.SUM:
                    agg_result = sum(values)
                elif aggregate_function == AggregateFunction.AVG:
                    agg_result = sum(values) / len(values)
                elif aggregate_function == AggregateFunction.MIN:
                    agg_result = min(values)
                elif aggregate_function == AggregateFunction.MAX:
                    agg_result = max(values)
            else:
                agg_result = 0 if aggregate_function == AggregateFunction.COUNT else None

            agg_col_name = f"{aggregate_function.name}({aggregate_column})"
            all_columns = columns.copy()
            if agg_col_name not in all_columns:
                all_columns.append(agg_col_name)

            for r in rows:
                row_data = {col: (r.id if col == "Id" else r.dimensions.get(col, r.facts.get(col))) for col in columns}
                row_data[agg_col_name] = agg_result
                result.append(row_data)
        else:
            all_columns = columns.copy()
            for r in rows:
                row_data = {col: (r.id if col == "Id" else r.dimensions.get(col, r.facts.get(col))) for col in columns}
                result.append(row_data)

        if raw_output:
            return result

        # Formatirani ispis
        if result:
            column_widths = {}
            for col in all_columns:
                max_len = len(str(col))
                for row in result:
                    val = row.get(col, "")
                    max_len = max(max_len, len(str(val)))
                column_widths[col] = max_len

            header = " | ".join(f"{col:<{column_widths[col]}}" for col in all_columns)
            separator = "-+-".join("-" * column_widths[col] for col in all_columns)
            rows_formatted = [" | ".join(f"{str(row.get(col,'')):<{column_widths[col]}}" for col in all_columns) for row in result]
            formatted_output = f"\n{header}\n{separator}\n" + "\n".join(rows_formatted)
            return formatted_output
        else:
            return "No rows returned."


    def execute_update(self, updates: Dict[str, Any], conditions: List[SearchCondition],
                       logical_op: LogicalOperator, use_index=True):
        query = SearchQuery(conditions=conditions, logical_op=logical_op, use_index=use_index)
        rows = self.search_with_index(query) if use_index else self.search_without_index(query)
        for r in rows:
            for col, val in updates.items():
                if col in r.facts:
                    r.facts[col] = val
                elif col in r.dimensions:
                    old_val = r.dimensions[col]
                    r.dimensions[col] = val
                    self._update_index_for_row(0, r)

    def execute_delete(self, conditions: List[SearchCondition],
                       logical_op: LogicalOperator, use_index=True):
        query = SearchQuery(conditions=conditions, logical_op=logical_op, use_index=use_index)
        rows = self.search_with_index(query) if use_index else self.search_without_index(query)
        for r in rows:
            self.delete_row(r.id)

# ----------------------------
# SQL PARSER
# ----------------------------

class SQLParser:
    @staticmethod
    def parse_where_clause(where_str: str):
        logical_op = LogicalOperator.AND
        if " OR " in where_str.upper():
            logical_op = LogicalOperator.OR
            conditions_strs = re.split(r"\s+OR\s+", where_str, flags=re.IGNORECASE)
        else:
            conditions_strs = re.split(r"\s+AND\s+", where_str, flags=re.IGNORECASE)

        conditions = []
        for cond_str in conditions_strs:
            m = re.match(r"\s*([A-Za-z0-9_]+)\s*=\s*('?)([^']+)\2\s*", cond_str.strip())
            if m:
                col = m.group(1)
                val = m.group(3)
                if val.isdigit():
                    val = int(val)
                else:
                    try:
                        val = float(val)
                    except:
                        pass
                conditions.append(SearchCondition(column=col, value=val))
        return conditions, logical_op

    @staticmethod
    def parse_select_columns(cols_str: str):
        cols = []
        agg_func = None
        agg_col = None
        for part in cols_str.split(","):
            part = part.strip()
            m = re.match(r"(SUM|AVG|MIN|MAX|COUNT)\((\w+)\)", part, flags=re.IGNORECASE)
            if m:
                agg_func = AggregateFunction[m.group(1).upper()]
                agg_col = m.group(2)
            else:
                cols.append(part)
        return cols, agg_func, agg_col

    @staticmethod
    def execute_sql(lsm: LSMIndexSystem, sql: str):
        sql = sql.strip().rstrip(";")
        sql_upper = sql.upper()

        if sql_upper.startswith("SELECT"):
            m = re.match(r"SELECT\s+(.+?)\s+FROM\s+(\w+)\s*(WHERE\s+(.+))?", sql, flags=re.IGNORECASE)
            if not m:
                raise ValueError("Nevalidan SELECT upit")
            cols_str = m.group(1).strip()
            table_name = m.group(2).strip()
            where_str = m.group(4)

            if table_name != "table":
                raise ValueError(f"Nepoznata tabela: {table_name}")

            columns, aggregate_function, aggregate_column = SQLParser.parse_select_columns(cols_str)
            if not columns and not aggregate_function:
                columns = lsm.dimension_columns + lsm.fact_columns

            conditions, logical_op = SQLParser.parse_where_clause(where_str) if where_str else ([], LogicalOperator.AND)
            return lsm.execute_select(
                columns=columns,
                conditions=conditions,
                logical_op=logical_op,
                aggregate_function=aggregate_function,
                aggregate_column=aggregate_column,
                benchmark=True
            )

        elif sql_upper.startswith("UPDATE"):
            m = re.match(r"UPDATE\s+(\w+)\s+SET\s+(.+?)\s*(WHERE\s+(.+))?$", sql, flags=re.IGNORECASE)
            if not m:
                raise ValueError("Nevalidan UPDATE upit")
            table_name = m.group(1).strip()
            set_str = m.group(2)
            where_str = m.group(4)

            if table_name != "table":
                raise ValueError(f"Nepoznata tabela: {table_name}")

            updates = {}
            for part in set_str.split(","):
                k, v = part.split("=")
                k = k.strip()
                v = v.strip().strip("'")
                if v.isdigit():
                    v = int(v)
                else:
                    try:
                        v = float(v)
                    except:
                        pass
                updates[k] = v

            conditions, logical_op = SQLParser.parse_where_clause(where_str) if where_str else ([], LogicalOperator.AND)
            lsm.execute_update(updates=updates, conditions=conditions, logical_op=logical_op)
            return f"UPDATE izvršen"

        elif sql_upper.startswith("DELETE"):
            m = re.match(r"DELETE\s+FROM\s+(\w+)\s*(WHERE\s+(.+))?$", sql, flags=re.IGNORECASE)
            if not m:
                raise ValueError("Nevalidan DELETE upit")
            table_name = m.group(1).strip()
            where_str = m.group(3)

            if table_name != "table":
                raise ValueError(f"Nepoznata tabela: {table_name}")

            conditions, logical_op = SQLParser.parse_where_clause(where_str) if where_str else ([], LogicalOperator.AND)
            lsm.execute_delete(conditions=conditions, logical_op=logical_op)
            return f"DELETE izvršen"

        else:
            raise ValueError("Nepoznata SQL komanda")

# ----------------------------
# CSV / JSON Učitavanje
# ----------------------------

def load_data_from_csv(filename: str) -> List[Dict[str, Any]]:
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                converted_row = {}
                for k, v in row.items():
                    try:
                        if k == 'Id':
                            converted_row[k] = int(v)
                        elif v.isdigit():
                            converted_row[k] = int(v)
                        else:
                            try:
                                converted_row[k] = float(v)
                            except:
                                converted_row[k] = v
                    except:
                        converted_row[k] = v
                data.append(converted_row)
    except Exception as e:
        print(f"Greška u čitanju CSV fajla {filename}: {e}")
    return data

def load_data_from_file(filename: str) -> List[Dict[str, Any]]:
    if filename.endswith('.csv'):
        return load_data_from_csv(filename)
    else:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Greška u učitavanju fajla {filename}: {e}")
            return []

# ----------------------------
# FORMATIRANJE REZULTATA
# ----------------------------

def format_results(results: list, columns: list) -> str:
    if not results:
        return "No rows returned."

    # Ako lista sadrži dict-ove, koristimo get
    column_widths = {}
    for col in columns:
        max_len = len(str(col))
        for row in results:
            if isinstance(row, dict):
                val = row.get(col, "")
            else:
                val = getattr(row, col, "")
            max_len = max(max_len, len(str(val)))
        column_widths[col] = max_len

    header = " | ".join(f"{col:<{column_widths[col]}}" for col in columns)
    separator = "-+-".join("-" * column_widths[col] for col in columns)
    rows_formatted = []
    for row in results:
        row_vals = []
        for col in columns:
            if isinstance(row, dict):
                val = row.get(col, "")
            else:
                val = getattr(row, col, "")
            row_vals.append(f"{str(val):<{column_widths[col]}}")
        rows_formatted.append(" | ".join(row_vals))

    return f"\n{header}\n{separator}\n" + "\n".join(rows_formatted)

# ----------------------------
# DEMO / TEST
# ----------------------------
if __name__ == "__main__":
    schema = {"dimensions": ["Region", "Product", "Year"], "facts": ["Sales", "Profit", "Quantity"]}
    lsm = LSMIndexSystem(schema)

    # Učitavanje sample podataka
    filename = "sample_data.csv"
    data_rows = load_data_from_csv(filename)

    for data in data_rows[:1000]:
        dimensions = {k: v for k, v in data.items() if k in schema["dimensions"]}
        facts = {k: v for k, v in data.items() if k in schema["facts"]}
        row_id = data.get("Id")
        lsm.insert_row(dimensions, facts, row_id=row_id)

    # --- SELECT sa i bez indeksa ---
    print("\n--- SQL SELECT primer ---")
    sql_select = "SELECT Id, Year, Region, Product, Sales, Profit, COUNT(Sales) FROM table WHERE Year=2024 AND Region='South';"
    res = SQLParser.execute_sql(lsm, sql_select)
    print(res)

    # Direktno SELECT bez indeksa
    print("\n--- SELECT bez indeksa (direktno) ---")
    res_no_index = lsm.execute_select(
        columns=["Id", "Year", "Region", "Product", "Sales", "Profit"],
        conditions=[SearchCondition("Year", 2024), SearchCondition("Region", "South")],
        logical_op=LogicalOperator.AND,
        use_index=False,
        raw_output=True,  # <--- važno
        benchmark=False
    )
    print(format_results(res_no_index, ["Id", "Year", "Region", "Product", "Sales", "Profit"]))

    # --- UPDATE primer ---
    print("\n--- SQL UPDATE primer ---")
    sql_update = "UPDATE table SET Profit=999 WHERE Region='South';"
    res = SQLParser.execute_sql(lsm, sql_update)
    print(res)

    # SELECT nakon UPDATE
    sql_select_after_update = "SELECT Region, Product, Sales, Profit FROM table WHERE Region='South' AND Profit=999;"
    res_after_update = SQLParser.execute_sql(lsm, sql_select_after_update)
    print(res_after_update)

    # --- DELETE primer ---
    print("\n--- SQL DELETE primer ---")
    sql_delete = "DELETE FROM table WHERE Product='GPU';"
    res = SQLParser.execute_sql(lsm, sql_delete)
    print(res)

    # SELECT nakon DELETE
    sql_select_after_delete = "SELECT Region, Product, Sales, Profit FROM table WHERE Product='GPU';"
    res_after_delete = SQLParser.execute_sql(lsm, sql_select_after_delete)
    print(res_after_delete)

    # --- Test dodavanja i brisanja reda ---
    print("\n--- Test dodavanja i brisanja reda ---")
    test_row_id = lsm.insert_row(
        dimensions={"Region": "TestRegion", "Product": "TestProd", "Year": 2024},
        facts={"Sales": 1234, "Profit": 567, "Quantity": 10}
    )

    sql_select_test = "SELECT Region, Product, Sales, Profit FROM table WHERE Region='TestRegion';"
    print("Pre brisanja:")
    print(SQLParser.execute_sql(lsm, sql_select_test))

    sql_delete_test = f"DELETE FROM table WHERE Id={test_row_id};"
    SQLParser.execute_sql(lsm, sql_delete_test)

    print("Posle brisanja:")
    print(SQLParser.execute_sql(lsm, sql_select_test))