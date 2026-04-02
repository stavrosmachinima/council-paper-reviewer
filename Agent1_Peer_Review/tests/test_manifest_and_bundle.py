from __future__ import annotations

import json
from pathlib import Path

from src.council.bundle import build_bundle
from src.council.manifest import load_manifest


def create_pdf(path: Path, text: str) -> None:
    path.write_bytes(f"%PDF-1.4\n{text}\n%%EOF".encode("utf-8"))


def test_manifest_and_bundle_discovery_excludes_private_assets_and_includes_external_evidence(tmp_path: Path) -> None:
    repo = tmp_path / "elsevier"
    mos_repo = tmp_path / "audiorate"
    (repo / "paper").mkdir(parents=True)
    (repo / "data").mkdir()
    (repo / "private").mkdir()
    (repo / "papers").mkdir()
    (repo / "scripts").mkdir()
    (mos_repo / "audiorate" / "public").mkdir(parents=True)
    (mos_repo / "node_modules").mkdir()
    (mos_repo / "logs").mkdir()
    (mos_repo / "data").mkdir()
    (mos_repo / ".git").mkdir()
    (mos_repo / "assets" / "js").mkdir(parents=True)
    (mos_repo / "audiorate" / "templates" / "public").mkdir(parents=True)

    (repo / "paper" / "main.tex").write_text(
        "\\section{Introduction}\nThis is the introduction.\n\n\\section{Methodology}\nMethods here.",
        encoding="utf-8",
    )
    create_pdf(repo / "paper" / "main.pdf", "Main manuscript PDF")
    (repo / "paper" / "references.bib").write_text("@article{sample,title={Sample}}\n", encoding="utf-8")
    (repo / "paper" / "elsarticle.cls").write_text("template", encoding="utf-8")
    (repo / "paper" / "elsarticle-num.bst").write_text("template", encoding="utf-8")
    (repo / "data" / "mos_summary_by_model.csv").write_text("model,mos\nfs2,3.1\n", encoding="utf-8")
    (repo / "data" / "dev.db").write_text("should never be loaded", encoding="utf-8")
    (repo / "scripts" / "compute_enhanced_stats.py").write_text("print('stats')\n", encoding="utf-8")
    (repo / "README.md").write_text("Repo overview", encoding="utf-8")
    (repo / "CLAUDE.md").write_text("Claude guidance", encoding="utf-8")
    (repo / "private" / "notes.txt").write_text("private", encoding="utf-8")
    create_pdf(repo / "papers" / "reference.pdf", "Reference PDF")
    create_pdf(repo / "elsevier_guide.pdf", "Guide for authors")
    (mos_repo / "README.md").write_text("AudioRate repo overview", encoding="utf-8")
    (mos_repo / "audiorate" / "public" / "views.py").write_text("print('views')\n", encoding="utf-8")
    (mos_repo / "audiorate" / "public" / "models.py").write_text("print('models')\n", encoding="utf-8")
    (mos_repo / "audiorate" / "public" / "forms.py").write_text("print('forms')\n", encoding="utf-8")
    (mos_repo / "audiorate" / "templates" / "public" / "home.html").write_text("<html>rate</html>\n", encoding="utf-8")
    (mos_repo / "assets" / "js" / "script.js").write_text("console.log('ui')\n", encoding="utf-8")
    (mos_repo / "data" / "models.json").write_text("{\"models\": []}\n", encoding="utf-8")
    (mos_repo / "data" / "samples.json").write_text("{\"samples\": []}\n", encoding="utf-8")
    (mos_repo / "package-lock.json").write_text("{\"packages\": {}}\n", encoding="utf-8")
    (mos_repo / "node_modules" / "junk.js").write_text("junk", encoding="utf-8")
    (mos_repo / "logs" / "audiorate.log").write_text("junk", encoding="utf-8")
    (mos_repo / "data" / "dev.db").write_text("junk", encoding="utf-8")
    (mos_repo / ".env").write_text("SECRET=1", encoding="utf-8")
    (mos_repo / ".git" / "config").write_text("junk", encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "target_repo": str(repo),
                "review_focus": ["stats", "narrative flow"],
                "external_evidence": [
                    {
                        "name": "audiorate",
                        "root": str(mos_repo),
                        "include_paths": [
                            "README.md",
                            "audiorate/public/views.py",
                            "audiorate/public/models.py",
                            "audiorate/public/forms.py",
                            "audiorate/templates/public/home.html",
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
                        "source_group": "mos_app",
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
    assert str(repo / "data" / "mos_summary_by_model.csv") in paths
    assert str(repo / "scripts" / "compute_enhanced_stats.py") in paths
    assert str(repo / "papers" / "reference.pdf") in paths
    assert str(mos_repo / "README.md") in paths
    assert str(mos_repo / "audiorate" / "public" / "views.py") in paths
    assert str(mos_repo / "audiorate" / "public" / "models.py") in paths
    assert str(mos_repo / "audiorate" / "public" / "forms.py") in paths
    assert str(mos_repo / "audiorate" / "templates" / "public" / "home.html") in paths
    assert str(mos_repo / "assets" / "js" / "script.js") in paths
    assert str(mos_repo / "data" / "models.json") in paths
    assert str(mos_repo / "data" / "samples.json") in paths
    assert str(repo / "data" / "dev.db") not in paths
    assert str(repo / "private" / "notes.txt") not in paths
    assert str(repo / "paper" / "elsarticle.cls") not in paths
    assert str(repo / "paper" / "elsarticle-num.bst") not in paths
    assert str(mos_repo / "package-lock.json") not in paths
    assert str(mos_repo / "node_modules" / "junk.js") not in paths
    assert str(mos_repo / "logs" / "audiorate.log") not in paths
    assert str(mos_repo / "data" / "dev.db") not in paths
    assert str(mos_repo / ".env") not in paths
