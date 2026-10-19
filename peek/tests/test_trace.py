"""Offline regression coverage for Python call graphs and trace queries."""

import pytest

from peek.scanner import scan
from peek.trace.builder import build_trace_graph
from peek.trace.models import FuncId
from peek.trace.query import find_by_location, trace


def build_fixture(root, files):
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return build_trace_graph(scan(root))


@pytest.fixture
def chain(tmp_path):
    graph = build_fixture(
        tmp_path,
        {"app.py": "def start():\n    middle()\ndef middle():\n    end()\ndef end():\n    pass\n"},
    )
    ids = {
        name: FuncId((tmp_path / "app.py").resolve(), name) for name in ("start", "middle", "end")
    }
    return graph, ids


def test_local_calls_have_forward_and_reverse_edges(chain):
    graph, ids = chain
    assert set(graph.nodes) == set(ids.values())
    assert {(e.caller, e.callee_resolved) for e in graph.edges} == {
        (ids["start"], ids["middle"]),
        (ids["middle"], ids["end"]),
    }
    edge = graph.edges_by_caller[ids["start"]][0]
    assert edge.callee_resolved == ids["middle"]
    assert graph.edges_by_callee[ids["middle"]] == [edge]
    assert edge.lineno == 2
    assert not edge.is_external


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("callees", {"end"}), ("callers", {"start"}), ("both", {"start", "end"})],
)
def test_query_directions(chain, direction, expected):
    graph, ids = chain
    tree = trace(graph, ids["middle"], direction=direction, depth=1)
    assert tree.root.fid == ids["middle"]
    assert {child.fid for child in tree.root.children} == {ids[name] for name in expected}
    assert tree.total_nodes == 1 + len(expected)
    assert all(not child.children for child in tree.root.children)


def test_depth_limit_stops_at_requested_hop(chain):
    graph, ids = chain
    shallow = trace(graph, ids["start"], depth=1)
    assert [child.fid for child in shallow.root.children] == [ids["middle"]]
    assert shallow.root.children[0].children == []
    deep = trace(graph, ids["start"], depth=2)
    assert deep.root.children[0].children[0].fid == ids["end"]
    assert deep.total_nodes == 3


def test_imported_call_and_cross_file_filter(tmp_path):
    graph = build_fixture(
        tmp_path,
        {
            "app.py": "from helpers import work\ndef run():\n    work()\n",
            "helpers.py": "def work():\n    pass\n",
        },
    )
    caller = FuncId((tmp_path / "app.py").resolve(), "run")
    callee = FuncId((tmp_path / "helpers.py").resolve(), "work")
    assert {(e.caller, e.callee_resolved) for e in graph.edges} == {(caller, callee)}
    assert trace(graph, caller).root.children[0].fid == callee
    assert trace(graph, callee, direction="callers").root.children[0].fid == caller
    assert trace(graph, caller, cross_file=False).root.children == []
    assert trace(graph, callee, direction="callers", cross_file=False).root.children == []


@pytest.mark.parametrize("direction", ["callees", "callers", "both"])
def test_recursive_cycles_terminate(tmp_path, direction):
    graph = build_fixture(
        tmp_path, {"recursive.py": "def first():\n    second()\ndef second():\n    first()\n"}
    )
    first = FuncId((tmp_path / "recursive.py").resolve(), "first")
    second = FuncId((tmp_path / "recursive.py").resolve(), "second")
    tree = trace(graph, first, direction=direction, depth=20)
    assert tree.root.children
    for child in tree.root.children:
        assert child.fid == second
        assert len(child.children) == 1
        cycle = child.children[0]
        assert cycle.fid == first
        assert cycle.is_cycle
        assert cycle.children == []
    assert tree.total_nodes == (5 if direction == "both" else 3)


def test_location_lookup_checks_file_and_function_range(chain):
    graph, ids = chain
    for query in ("app.py", str(ids["middle"].file)):
        assert find_by_location(graph, query, 3) == ids["middle"]
        assert find_by_location(graph, query, 4) == ids["middle"]
    assert find_by_location(graph, "missing.py", 4) is None
    # Lines outside a function intentionally fall back to the nearest preceding def.
    assert find_by_location(graph, "app.py", 99) == ids["end"]
    assert find_by_location(graph, "app.py", 0) is None


def test_malformed_and_empty_files_do_not_hide_valid_functions(tmp_path):
    graph = build_fixture(
        tmp_path,
        {"broken.py": "def broken(:\n", "empty.py": "", "valid.py": "def valid():\n    pass\n"},
    )
    valid = FuncId((tmp_path / "valid.py").resolve(), "valid")
    assert set(graph.nodes) == {valid}
    assert graph.edges == []
    assert trace(graph, valid).total_nodes == 1


@pytest.mark.parametrize("files", [{}, {"broken.py": "def broken(:\n", "empty.py": ""}])
def test_empty_or_unparseable_repository_has_no_graph_nodes(tmp_path, files):
    graph = build_fixture(tmp_path, files)
    assert graph.nodes == {}
    assert graph.edges == []
    assert graph.warnings
