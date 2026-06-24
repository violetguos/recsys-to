from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class CheckResult:
    name: str
    description: str
    passed: bool
    details: dict[str, Any]
    blocking: bool = False

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name}\n"
            f"  Description: {self.description}\n"
            f"  Blocking: {'YES' if self.blocking else 'NO'}\n"
            f"  Details: {self.details}\n"
        )


class DataQualityChecker:
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"

    SCHEMAS: dict[str, dict[str, Any]] = {
        "aisles": {
            "columns": ["aisle_id", "aisle"],
        },
        "departments": {
            "columns": ["department_id", "department"],
        },
        "products": {
            "columns": ["product_id", "product_name", "aisle_id", "department_id"],
        },
        "orders": {
            "columns": [
                "order_id", "user_id", "eval_set", "order_number",
                "order_dow", "order_hour_of_day", "days_since_prior_order",
            ],
        },
        "order_products__prior": {
            "columns": ["order_id", "product_id", "add_to_cart_order", "reordered"],
        },
        "order_products__train": {
            "columns": ["order_id", "product_id", "add_to_cart_order", "reordered"],
        },
    }

    TABLE_NAMES = [
        "aisles", "departments", "products", "orders",
        "order_products__prior", "order_products__train",
    ]

    def __init__(self, sample_size: int | None = None):
        self.results: list[CheckResult] = []
        self.sample_size = sample_size
        self.tables: dict[str, pd.DataFrame] = {}
        self.load_errors: dict[str, str] = {}
        self._load_all()

    def _load_all(self) -> None:
        for name in self.TABLE_NAMES:
            path = self.DATA_DIR / f"{name}.csv"
            try:
                df = pd.read_csv(path, low_memory=False, nrows=self.sample_size)
                self.tables[name] = df
            except Exception as e:
                self.load_errors[name] = str(e)

        if "order_products__prior" in self.tables and "order_products__train" in self.tables:
            self.tables["order_products"] = pd.concat(
                [self.tables["order_products__prior"], self.tables["order_products__train"]],
                ignore_index=True,
            )

    # ------------------------------------------------------------------
    # 1. Schema Validation
    # ------------------------------------------------------------------
    def check_schema_validation(self) -> None:
        for name in self.TABLE_NAMES:
            expected = self.SCHEMAS[name]["columns"]
            if name in self.load_errors:
                self._add_result(
                    f"schema_validation:{name}",
                    "Verify file exists and columns match the expected schema",
                    False,
                    {"error": self.load_errors[name]},
                    blocking=True,
                )
                continue
            df = self.tables[name]
            actual = list(df.columns)
            missing = [c for c in expected if c not in actual]
            extra = [c for c in actual if c not in expected]
            passed = len(missing) == 0 and len(extra) == 0 and len(self.load_errors) == 0
            self._add_result(
                f"schema_validation:{name}",
                f"Verify {name}.csv has expected columns: {expected}",
                passed,
                {"expected_columns": expected, "actual_columns": actual, "missing": missing, "extra": extra},
                blocking=True,
            )

    # ------------------------------------------------------------------
    # 2. Missing Identifiers
    # ------------------------------------------------------------------
    def check_missing_identifiers(self) -> None:
        id_columns = {
            "aisles": ["aisle_id"],
            "departments": ["department_id"],
            "products": ["product_id", "aisle_id", "department_id"],
            "orders": ["order_id", "user_id"],
            "order_products__prior": ["order_id", "product_id"],
            "order_products__train": ["order_id", "product_id"],
        }
        for name, id_cols in id_columns.items():
            if name in self.load_errors:
                continue
            df = self.tables[name]
            nulls = {col: int(df[col].isna().sum()) for col in id_cols if col in df.columns}
            total_nulls = sum(nulls.values())
            passed = total_nulls == 0
            self._add_result(
                f"missing_identifiers:{name}",
                f"Check for null values in primary/foreign key columns: {id_cols}",
                passed,
                {"null_counts": nulls, "total_nulls": total_nulls, "table": name},
                blocking=True,
            )

    # ------------------------------------------------------------------
    # 3. Duplicate Records
    # ------------------------------------------------------------------
    def check_duplicate_records(self) -> None:
        pk_map = {
            "aisles": ["aisle_id"],
            "departments": ["department_id"],
            "products": ["product_id"],
            "orders": ["order_id"],
            "order_products__prior": ["order_id", "product_id"],
            "order_products__train": ["order_id", "product_id"],
        }
        for name, pk in pk_map.items():
            if name in self.load_errors:
                continue
            df = self.tables[name]
            total = len(df)
            dupes = int(df.duplicated(subset=pk, keep=False).sum())
            passed = dupes == 0
            dupe_pct = round(dupes / total * 100, 4) if total > 0 else 0.0
            self._add_result(
                f"duplicate_records:{name}",
                f"Check for duplicate rows based on primary key {pk}",
                passed,
                {"table": name, "total_rows": total, "duplicate_rows": dupes, "duplicate_pct": dupe_pct},
                blocking=False,
            )

    # ------------------------------------------------------------------
    # 4. Invalid Timestamps
    # ------------------------------------------------------------------
    def check_invalid_timestamps(self) -> None:
        if "orders" in self.load_errors:
            return
        df = self.tables["orders"]
        issues: dict[str, Any] = {}

        dow_vals = df["order_dow"]
        bad_dow = int(dow_vals.isin(range(0, 7)).__invert__().sum())
        if bad_dow:
            issues["order_dow_out_of_range"] = {
                "count": bad_dow,
                "pct": round(bad_dow / len(df) * 100, 4),
                "range": "0-6",
                "example_values": dow_vals[~dow_vals.isin(range(0, 7))].dropna().unique()[:5].tolist(),
            }

        hour_vals = df["order_hour_of_day"]
        bad_hour = int(hour_vals.isin(range(0, 24)).__invert__().sum())
        if bad_hour:
            issues["order_hour_of_day_out_of_range"] = {
                "count": bad_hour,
                "pct": round(bad_hour / len(df) * 100, 4),
                "range": "0-23",
                "example_values": hour_vals[~hour_vals.isin(range(0, 24))].dropna().unique()[:5].tolist(),
            }

        dspo_vals = df["days_since_prior_order"]
        bad_dspo = int((dspo_vals < 0).sum())
        if bad_dspo:
            issues["days_since_prior_order_negative"] = {
                "count": bad_dspo,
                "pct": round(bad_dspo / len(df) * 100, 4),
                "example_values": dspo_vals[dspo_vals < 0].dropna().unique()[:5].tolist(),
            }

        passed = len(issues) == 0
        self._add_result(
            "invalid_timestamps:orders",
            "Validate order_dow (0-6), order_hour_of_day (0-23), days_since_prior_order (>=0 or NaN)",
            passed,
            issues if issues else {"result": "no issues found"},
            blocking=True,
        )

    # ------------------------------------------------------------------
    # 5. Invalid Interaction Values
    # ------------------------------------------------------------------
    def check_invalid_interaction_values(self) -> None:
        for table in ["order_products__prior", "order_products__train", "order_products"]:
            if table in self.load_errors:
                continue
            if table not in self.tables:
                continue

            df = self.tables[table]
            issues: dict[str, Any] = {}
            total = len(df)

            bad_add = int((df["add_to_cart_order"] < 1).sum())
            if bad_add:
                issues["add_to_cart_order_invalid"] = {
                    "count": bad_add,
                    "pct": round(bad_add / total * 100, 4),
                    "expected": ">= 1",
                    "example_values": df.loc[df["add_to_cart_order"] < 1, "add_to_cart_order"].unique()[:5].tolist(),
                }

            bad_reorder = int((~df["reordered"].isin([0, 1])).sum())
            if bad_reorder:
                issues["reordered_invalid"] = {
                    "count": bad_reorder,
                    "pct": round(bad_reorder / total * 100, 4),
                    "expected": "{0, 1}",
                    "example_values": df.loc[~df["reordered"].isin([0, 1]), "reordered"].unique()[:5].tolist(),
                }

            passed = len(issues) == 0
            display_name = table
            self._add_result(
                f"invalid_interaction_values:{display_name}",
                "Validate add_to_cart_order (>=1) and reordered ({0, 1})",
                passed,
                issues if issues else {"result": "no issues found", "total_rows": total},
                blocking=True,
            )

        if "orders" not in self.load_errors:
            odf = self.tables["orders"]
            bad_ord_num = int((odf["order_number"] < 1).sum())
            passed = bad_ord_num == 0
            self._add_result(
                "invalid_interaction_values:orders",
                "Validate order_number (>= 1)",
                passed,
                {"invalid_order_number_count": bad_ord_num} if bad_ord_num else {"result": "no issues found"},
                blocking=True,
            )

    # ------------------------------------------------------------------
    # 6. Reference Integrity
    # ------------------------------------------------------------------
    def check_reference_integrity(self) -> None:
        data_dir = self.DATA_DIR

        def load_keys(filename: str, column: str) -> set[Any]:
            try:
                return set(pd.read_csv(data_dir / filename, usecols=[column], low_memory=False).iloc[:, 0])
            except Exception:
                return set()

        violations: dict[str, Any] = {}

        aisle_ids = load_keys("aisles.csv", "aisle_id") if "aisles" not in self.load_errors else set()
        dept_ids = load_keys("departments.csv", "department_id") if "departments" not in self.load_errors else set()
        product_ids = load_keys("products.csv", "product_id") if "products" not in self.load_errors else set()
        order_ids = load_keys("orders.csv", "order_id") if "orders" not in self.load_errors else set()
        op_prior_ids = load_keys("order_products__prior.csv", "product_id") if "order_products__prior" not in self.load_errors else set()
        op_train_ids = load_keys("order_products__train.csv", "product_id") if "order_products__train" not in self.load_errors else set()
        op_order_ids = load_keys("order_products__prior.csv", "order_id") | load_keys("order_products__train.csv", "order_id")
        op_product_ids = op_prior_ids | op_train_ids

        products = self.tables.get("products")
        if products is not None and aisle_ids:
            product_aisles = set(products["aisle_id"].unique())
            orphan = product_aisles - aisle_ids
            if orphan:
                violations["products->aisles"] = {
                    "description": "products.aisle_id references aisles.aisle_id",
                    "orphan_count": len(orphan),
                    "example_orphan_ids": sorted(orphan)[:10],
                }

        if products is not None and dept_ids:
            product_depts = set(products["department_id"].unique())
            orphan = product_depts - dept_ids
            if orphan:
                violations["products->departments"] = {
                    "description": "products.department_id references departments.department_id",
                    "orphan_count": len(orphan),
                    "example_orphan_ids": sorted(orphan)[:10],
                }

        if op_product_ids and product_ids:
            orphan = op_product_ids - product_ids
            if orphan:
                violations["order_products->products"] = {
                    "description": "order_products.product_id references products.product_id",
                    "orphan_count": len(orphan),
                    "example_orphan_ids": sorted(orphan)[:10],
                }

        if op_order_ids and order_ids:
            orphan = op_order_ids - order_ids
            if orphan:
                violations["order_products->orders"] = {
                    "description": "order_products.order_id references orders.order_id",
                    "orphan_count": len(orphan),
                    "example_orphan_ids": sorted(orphan)[:10],
                }

        passed = len(violations) == 0
        self._add_result(
            "reference_integrity",
            "Verify foreign key relationships between tables",
            passed,
            violations if violations else {"result": "all foreign key relationships are valid"},
            blocking=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_result(
        self, name: str, description: str, passed: bool,
        details: dict[str, Any], blocking: bool = False,
    ) -> None:
        self.results.append(CheckResult(
            name=name, description=description, passed=passed,
            details=details, blocking=blocking,
        ))

    def run_all(self) -> list[CheckResult]:
        self.results.clear()
        self.check_schema_validation()
        self.check_missing_identifiers()
        self.check_duplicate_records()
        self.check_invalid_timestamps()
        self.check_invalid_interaction_values()
        self.check_reference_integrity()
        return self.results

    def print_report(self, file=sys.stdout) -> None:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        header = f"{'='*60}\n  Data Quality Report\n{'='*60}\n"
        file.write(header)
        for r in self.results:
            file.write(str(r))
        footer = (
            f"{'='*60}\n"
            f"  Results: {passed}/{total} checks passed\n"
            f"{'='*60}\n"
        )
        file.write(footer)

    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def blocking_failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.blocking]

    def non_blocking_failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and not r.blocking]


def run_quality_checks(sample_size: int | None = None) -> bool:
    checker = DataQualityChecker(sample_size=sample_size)
    checker.run_all()
    checker.print_report()
    return checker.all_passed()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run data quality checks on Instacart dataset")
    parser.add_argument("--sample", type=int, default=None, help="Number of rows to sample per table")
    args = parser.parse_args()
    ok = run_quality_checks(sample_size=args.sample)
    sys.exit(0 if ok else 1)
