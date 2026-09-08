# third_party/

Vendored upstream code that ships **no Python packaging**, so it cannot be
`pip install`ed from PyPI and must be made importable locally.

## Convention

A vendored dependency is **a git submodule** — the upstream source, pinned to a
commit (see `../.gitmodules`). E.g. `joyai_echo/`, `videophy/`.

A single **editable-install wrapper**, `third_party/pyproject.toml`, exposes
every submodule's un-packaged `src` tree as a real importable package via
`[tool.setuptools.packages.find]` (`where` = the src roots, `include` = the
package names). `pip install -e third_party` makes them all importable, so the
main repo (`vrl/`) needs no `sys.path` injection. `make setup` (repo root) runs
this for you after fetching the submodules.

## Current vendored packages

| submodule        | exposes                                                       |
| ---------------- | ------------------------------------------------------------ |
| `joyai_echo`     | `ltx_core`, `ltx_pipelines`, `ltx_distillation`              |
| `videophy`       | `mplug_owl_video`                                            |
| `PhyMotion`      | _(not imported — run via CLI)_ `astrolabe.rewards` via `vrl/scripts/eval/phymotion_score.py` |
| `VMBench`        | _(not imported — run via CLI)_ motion-eval benchmark; fold scores in with `--merge-json` |
| `DynamicEval`    | _(not imported — run via CLI)_ dynamic-scene eval; fold scores in with `--merge-json` |
| `CausVid`        | `causvid` causal-Wan model/runtime (in-process; released weights are non-commercial) |
| `MAGI-1`         | _(not imported — isolated subprocess)_ official causal-chunk video generator |
| `vdn-minimax-h3` | `src` — VDN-H3 hybrid window-softmax / linear attention (see the name note below) |

The causal-chunk adapters enforce their audited source revisions at runtime:
`CausVid@adb6a5ecd07666b4d0290042915c8406e6d5ce22` and
`MAGI-1@0fcefdef8ce2df37a3b8890979433c06eb003328`. CausVid's source and
released generator checkpoint are CC BY-NC-SA 4.0 / non-commercial (the
checkpoint revision is separately pinned in the model preset); MAGI-1 source
and weights are Apache-2.0.

Not every vendored repo is exposed through `third_party/pyproject.toml`: the
wrapper lists only submodules that `vrl/` **imports** in-process. The three
motion-eval benchmarks above are invoked as external commands (their own CLIs,
or the PhyMotion bridge run in PhyMotion's own conda env), so they are vendored
to pin the code but stay out of the editable install — `make setup` simply
skips them.

## Adding a new vendored dependency

```bash
git submodule add <url> third_party/<name>
# In third_party/pyproject.toml: add the submodule's src root(s) to
#   [tool.setuptools.packages.find].where  and the package name(s) to .include
# In .gitignore: add `!third_party/<name>` (pyproject.toml is already whitelisted)
make setup
```

## The `vdn-minimax-h3` package name

VDN-H3 names its own top-level package `src` (its `pyproject.toml` declares
`packages.find include = ["src*"]`), so exposing the submodule verbatim means
exposing that generic name. It is kept rather than renamed because renaming the
directory would break the upstream tree's own absolute `from src.models...`
imports, and a verbatim pinned submodule is the point: the hybrid attention is
novel math that must not be transcribed into `vrl/`.

Two containments: the wrapper's `include` whitelists `src*` from that root only,
and `vrl/` funnels every use through one module,
`vrl/models/families/vdn_h3/vendor.py`. Grep that file to find every upstream
entry point VRL depends on.
