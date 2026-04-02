from __future__ import annotations

import json
from pathlib import Path

from src.council.bundle import build_bundle
from src.council.manifest import load_manifest


def create_pdf(path: Path, text: str) -> None:
    path.write_bytes(f"%PDF-1.4\n{text}\n%%EOF".encode("utf-8"))


def test_manifest_and_bundle_discovery_excludes_private_assets_and_includes_external_evidence(tmp_path: Path) -> None:
    repo = tmp_path / "paper_repo"
    eval_app = tmp_path / "eval_app"
    (repo / "paper").mkdir(parents=True)
    (repo / "data").mkdir()
    (repo / "private").mkdir()
    (repo / "papers").mkdir()
    (repo / "scripts").mkdir()
    (eval_app / "app" / "public").mkdir(parents=True)
    (eval_app / "node_modules").mkdir()
    (eval_app / "logs").mkdir()
    (eval_app / "data").mkdir()
    (eval_app / ".git").mkdir()
    (eval_app / "assets" / "js").mkdir(parents=True)
    (eval_app / "app" / "templates" / "public").mkdir(parents=True)

    (repo / "paper" / "main.tex").write_text(
        "\\section{Introduction}\nThis is the introduction.\n\n\\section{Methodology}\nMethods here.",
        encoding="utf-8",
    )
    create_pdf(repo / "paper" / "main.pdf", "Main manuscript PDF")
    (repo / "paper" / "references.bib").write_text("@article{sample,title={Sample}}\n", encoding="utf-8")
    (repo / "paper" / "elsarticle.cls").write_text("template", encoding="utf-8")
    (repo / "paper" / "elsarticle-num.bst").write_text("template", encoding="utf-8")
    (repo / "data" / "results_summary.csv").write_text("model,score\nA,3.1\n", encoding="utf-8")
    (repo / "data" / "dev.db").write_text("should never be loaded", encoding="utf-8")
    (repo / "scripts" / "compute_stats.py").write_text("print('stats')\n", encoding="utf-8")
    (repo / "README.md").write_text("Repo overview", encoding="utf-8")
    (repo / "CLAUDE.md").write_text("Claude guidance", encoding="utf-8")
    (repo / "private" / "notes.txt").write_text("private", encoding="utf-8")
    create_pdf(repo / "papers" / "reference.pdf", "Reference PDF")
    create_pdf(repo / "guide.pdf", "Guide for authors")
    (eval_app / "README.md").write_text("External evaluation app", encoding="utf-8")
    (eval_app / "app" / "public" / "views.py").write_text("print('views')\n", encoding="utf-8")
    (eval_app / "app" / "public" / "models.py").write_text("print('models')\n", encoding="utf-8")
    (eval_app / "app" / "public" / "forms.py").write_text("print('forms')\n", encoding="utf-8")
    (eval_app / "app" / "templates" / "public" / "home.html").write_text("<html>rate</html>\n", encoding="utf-8")
    (eval_app / "assets" / "js" / "script.js").write_text("console.log('ui')\n", encoding="utf-8")
    (eval_app / "data" / "models.json").write_text("{\"models\": []}\n", encoding="utf-8")
    (eval_app / "data" / "samples.json").write_text("{\"samples\": []}\n", encoding="utf-8")
    (eval_app / "package-lock.json").write_text("{\"packages\": {}}\n", encoding="utf-8")
    (eval_app / "node_modules" / "junk.js").write_text("junk", encoding="utf-8")
    (eval_app / "logs" / "app.log").write_text("junk", encoding="utf-8")
    (eval_app / "data" / "dev.db").write_text("junk", encoding="utf-8")
    (eval_app / ".env").write_text("SECRET=1", encoding="utf-8")
    (eval_app / ".git" / "config").write_text("junk", encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "target_repo": str(repo),
                "review_focus": ["stats", "narrative flow"],
                "external_evidence": [
                    {
                        "name": "eval_app",
                        "root": str(eval_app),
                        "include_paths": [
                            "README.md",
                            "app/public/views.py",
                            "app/public/models.py",
                            "app/public/forms.py",
                            "app/templates/public/home.html",
                            "assets/js/script.js",
                            "data/models.json",
                            "data/samples.json",
                        ],
                        "include_globs": [],
                        "exclude_globs": [
                            ".git/**",
                            "node_modules/**",
                            "logs/**",
                            ".env",
                            ".env.*",
                            "data/dev.db",
                            "__pycache__/**",
                            "*.pyc",
                        ],
                        "source_group": "external_app",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)
    bundle = build_bundle(manifest)
    paths = {document.path for document in bundle["documents"]}

    assert manifest.report_formats == ["json", "md"]
    assert str(repo / "paper" / "main.tex") in paths
    assert str(repo / "paper" / "main.pdf") in paths
    assert str(repo / "data" / "results_summary.csv") in paths
    assert str(repo / "scripts" / "compute_stats.py") in paths
    assert str(repo / "papers" / "reference.pdf") in paths
    assert str(eval_app / "README.md") in paths
    assert str(eval_app / "app" / "public" / "views.py") in paths
    assert str(eval_app / "app" / "public" / "models.py") in paths
    assert str(eval_app / "app" / "public" / "forms.py") in paths
    assert str(eval_app / "app" / "templates" / "public" / "home.html") in paths
    assert str(eval_app / "assets" / "js" / "script.js") in paths
    assert str(eval_app / "data" / "models.json") in paths
    assert str(eval_app / "data" / "samples.json") in paths
    assert str(repo / "data" / "dev.db") not in paths
    assert str(repo / "private" / "notes.txt") not in paths
    assert str(repo / "paper" / "elsarticle.cls") not in paths
    assert str(repo / "paper" / "elsarticle-num.bst") not in paths
    assert str(eval_app / "package-lock.json") not in paths
    assert str(eval_app / "node_modules" / "junk.js") not in paths
    assert str(eval_app / "logs" / "app.log") not in paths
    assert str(eval_app / "data" / "dev.db") not in paths
    assert str(eval_app / ".env") not in paths
