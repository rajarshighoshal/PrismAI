"""Static route-wiring checks for tool-server/main.py.

No service deps, no network. Catches decorator-placement regressions where a
FastAPI route accidentally lands on a helper instead of the intended handler.

Run from repo root:
  python3 tool-server/test_routes_static.py
"""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
tree = ast.parse((ROOT / "main.py").read_text())
fails = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}: {name}")
    if not cond:
        fails.append(name)


def route_decorators(fn: ast.FunctionDef) -> list[tuple[str, str]]:
    out = []
    for dec in fn.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        f = dec.func
        if not (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id == "app"
            and f.attr in {"get", "post", "put", "delete"}
        ):
            continue
        path = dec.args[0].value if dec.args and isinstance(dec.args[0], ast.Constant) else ""
        out.append((f.attr, path))
    return out


routes_by_fn = {
    node.name: route_decorators(node)
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}

check("lookup_doi_citation is the /lookup_doi_citation route",
      ("post", "/lookup_doi_citation") in routes_by_fn.get("lookup_doi_citation", []))
check("_title_similarity is not exposed as an API route",
      not routes_by_fn.get("_title_similarity"))
check("search_citation route still wired",
      ("post", "/search_citation") in routes_by_fn.get("search_citation", []))

print("\n" + ("all static route tests passed" if not fails else f"{len(fails)} FAILED: {fails}"))
raise SystemExit(1 if fails else 0)
