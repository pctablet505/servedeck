"""The documentation site (docs/, docs-theme/, mkdocs.yml): what it says is true, what it ships is well formed.

The first group reads only the repository: every page has a unique title and a description a search result can
show, the nav lists exactly the pages the site builds, and the public summaries (README, home page, llms.txt,
citation, package metadata) do not claim what the code contradicts: zero dependencies, zero-downtime switching,
or a client ``servedeck wire`` does not write.

The second group builds the site with ``mkdocs build --strict`` and reads the HTML. It needs mkdocs (the ``docs``
extra) and is skipped without it; .github/workflows/pages.yml installs it, so the deploy is gated on them.
"""

from __future__ import annotations

import argparse
import html
import html.parser
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from servedeck import cli, settings, wire

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
SITE_URL = "https://servedeck.rlalphalabs.com/"
OFF_SITE = ("imported-2026-09",)
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
TITLE_SUFFIX = " | Servedeck"

#: Summaries a stranger reads first. The deep docs may discuss anything; these may not overclaim.
PUBLIC_SUMMARIES = ("README.md", "docs/index.md", "docs/llms.txt", "CITATION.cff", "pyproject.toml", "mkdocs.yml")
BANNED = {
    "zero dependencies (fastapi, uvicorn, httpx and tomli-w are required)": r"zero[- ]?dependenc",
    "zero-downtime (a switch stops the model, waits for the GPU, then boots the next)": r"(?<!not )(?<!Not )zero[- ]?downtime",
    "hot swap (same reason)": r"hot[- ]?swap",
    "Cursor (servedeck wire does not write Cursor's config)": r"\bcursor\b",
    "servedeck.toml (no module reads it)": r"servedeck\.toml",
}


def source_pages() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md") if not any(part in OFF_SITE for part in p.relative_to(DOCS).parts))


def page_path(md: Path) -> str:
    """The lower-case address a page is served at, relative to the site root: ``OPERATIONS.md`` -> ``operations/``."""
    rel = md.relative_to(DOCS).with_suffix("").as_posix()
    return "" if rel == "index" else rel.lower() + "/"


def front_matter(md: Path) -> dict[str, str]:
    match = FRONT_MATTER.match(md.read_text(encoding="utf-8"))
    assert match, f"{md.relative_to(REPO)} has no front matter"
    return dict(re.findall(r'^(title|description): "([^"]*)"$', match.group(1), re.M))


def pyproject_version() -> str:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


class TestEveryPageCanBeFoundAndShown:
    def test_each_page_has_a_unique_title_and_a_description_that_fits_a_search_result(self):
        pages = source_pages()
        assert len(pages) >= 12, "the site's page set shrank"
        titles, descriptions = {}, {}
        for md in pages:
            fm = front_matter(md)
            name = md.relative_to(REPO).as_posix()
            assert set(fm) == {"title", "description"}, f"{name}: front matter needs a title and a description"
            full = fm["title"] if md.name == "index.md" else fm["title"] + TITLE_SUFFIX
            assert len(full) <= 70, f"{name}: title is {len(full)} characters, search results cut it near 60"
            assert 50 <= len(fm["description"]) <= 160, f"{name}: description is {len(fm['description'])} characters"
            assert titles.setdefault(fm["title"].lower(), name) == name, f"{name} repeats the title of {titles[fm['title'].lower()]}"
            assert descriptions.setdefault(fm["description"].lower(), name) == name, f"{name} repeats a description"

    def test_the_nav_lists_exactly_the_pages_that_exist(self):
        nav = (REPO / "mkdocs.yml").read_text(encoding="utf-8").split("\nnav:\n", 1)[1].split("\n\n", 1)[0]
        listed = set(re.findall(r":\s+(\S+\.md)\s*$", nav, re.M))
        existing = {p.relative_to(DOCS).as_posix() for p in source_pages()}
        assert listed == existing

    def test_llms_txt_links_every_page_of_the_site_and_nothing_else_on_it(self):
        text = (DOCS / "llms.txt").read_text(encoding="utf-8")
        linked = {u for u in re.findall(r"\((https://[^)]+)\)", text) if u.startswith(SITE_URL)}
        assert linked == {SITE_URL + page_path(p) for p in source_pages()}
        assert set(re.findall(r"\((https://[^)]+)\)", text)) - linked == {"https://github.com/pctablet505/servedeck"}

    def test_robots_txt_allows_crawling_and_names_the_sitemap(self):
        text = (DOCS / "robots.txt").read_text(encoding="utf-8")
        assert re.search(r"^User-agent: \*\nAllow: /$", text, re.M)
        assert f"Sitemap: {SITE_URL}sitemap.xml" in text
        assert (DOCS / "CNAME").read_text(encoding="utf-8").strip() == SITE_URL.split("//")[1].rstrip("/")


class TestWhatThePublicPagesClaimIsTrue:
    @pytest.mark.parametrize("claim", sorted(BANNED))
    def test_no_summary_makes_a_claim_the_code_contradicts(self, claim):
        offenders = [
            name for name in PUBLIC_SUMMARIES
            if re.search(BANNED[claim], (REPO / name).read_text(encoding="utf-8"), re.I if "Cursor" in claim else 0)
        ]
        assert not offenders, f"{claim}: {offenders}"

    def test_every_command_the_pages_show_is_a_real_subcommand(self):
        verbs = next(a for a in cli.build_parser()._actions if isinstance(a, argparse._SubParsersAction)).choices
        for name in ("README.md", "docs/index.md"):
            shown = set(re.findall(r"^servedeck (\w+)", (REPO / name).read_text(encoding="utf-8"), re.M))
            assert shown and shown <= set(verbs), f"{name} shows {sorted(shown - set(verbs))}, which the CLI does not have"

    def test_the_clients_named_are_the_clients_servedeck_wire_writes(self):
        writers = {n.removeprefix("render_") for n in wire.__all__ if n.startswith("render_")}
        assert writers == {"codex", "kimi", "vscode"}, "a client was added or removed: update the pages that name them"
        home = (DOCS / "index.md").read_text(encoding="utf-8")
        assert all(name in home for name in ("Codex CLI", "Kimi CLI", "VS Code"))

    def test_the_address_shown_is_the_address_it_listens_on(self):
        for name in ("README.md", "docs/index.md", "docs/llms.txt"):
            ports = set(re.findall(r"127\.0\.0\.1:(\d+)", (REPO / name).read_text(encoding="utf-8")))
            assert ports == {str(settings.DEFAULT_LISTEN_PORT)}, f"{name}: {ports}"

    def test_the_unit_file_the_home_page_tells_you_to_install_exists(self):
        assert "systemd/servedeck.service" in (DOCS / "index.md").read_text(encoding="utf-8")
        assert (REPO / "systemd" / "servedeck.service").is_file()

    def test_the_version_is_the_same_in_the_package_the_citation_and_the_page_markup(self):
        version = pyproject_version()
        assert re.search(rf'^version: "{re.escape(version)}"$', (REPO / "CITATION.cff").read_text(encoding="utf-8"), re.M)
        assert f'"version":"{version}"' in (REPO / "docs-theme" / "main.html").read_text(encoding="utf-8")


class _Page(html.parser.HTMLParser):
    """What a crawler reads from one page: the head's tags, the headings, the structured data and every resource URL."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.canonicals: list[str] = []
        self.meta: dict[str, str] = {}
        self.title = ""
        self.h1 = 0
        self.resources: list[str] = []
        self.jsonld: list[str] = []
        self._in = ""
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "link" and a.get("rel") == "canonical":
            self.canonicals.append(a.get("href", ""))
        elif tag == "link" and a.get("href"):
            self.resources.append(a["href"])
        elif tag == "meta":
            self.meta[a.get("name") or a.get("property") or a.get("http-equiv", "")] = a.get("content", "")
        elif tag == "h1":
            self.h1 += 1
        elif tag in ("script", "img", "iframe", "source") and a.get("src"):
            self.resources.append(a["src"])
        if tag == "script" and a.get("type") == "application/ld+json":
            self._in = "jsonld"
            self.jsonld.append("")
        elif tag == "title":
            self._in = "title"

    def handle_endtag(self, tag):
        self._in = ""

    def handle_data(self, data):
        if self._in == "title":
            self.title += data
        elif self._in == "jsonld":
            self.jsonld[-1] += data


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict[str, tuple[_Page, Path]]:
    pytest.importorskip("mkdocs")
    out = tmp_path_factory.mktemp("site")
    done = subprocess.run([sys.executable, "-m", "mkdocs", "build", "--strict", "-d", str(out)], cwd=REPO, capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout + done.stderr)[-3000:]
    pages = {}
    for p in out.rglob("index.html"):
        rel = p.parent.relative_to(out).as_posix()
        pages["" if rel == "." else rel + "/"] = (_Page(p.read_text(encoding="utf-8")), p)
    pages["__root__"] = (None, out)
    return pages


class TestTheBuiltSite:
    def _pages(self, built):
        return {k: v[0] for k, v in built.items() if k != "__root__"}

    def test_the_site_serves_exactly_the_pages_in_lower_case(self, built):
        assert set(self._pages(built)) == {page_path(p) for p in source_pages()}

    def test_each_page_has_one_canonical_that_is_its_own_address_and_one_h1(self, built):
        for path, page in self._pages(built).items():
            assert page.canonicals == [SITE_URL + path], path
            assert page.meta["og:url"] == SITE_URL + path, path
            assert page.h1 == 1, f"{path or 'home'} has {page.h1} h1 headings"
            assert page.meta["robots"].startswith("index, follow"), path

    def test_each_title_and_description_is_the_pages_own_and_no_two_titles_repeat(self, built):
        titles = []
        by_path = {page_path(p): front_matter(p) for p in source_pages()}
        for path, page in self._pages(built).items():
            fm = by_path[path]
            assert html.unescape(page.title) == (fm["title"] if path == "" else fm["title"] + TITLE_SUFFIX), path
            assert html.unescape(page.meta["description"]) == fm["description"], path
            titles.append(page.title)
        assert len(set(titles)) == len(titles)

    def test_no_page_loads_anything_from_another_host(self, built):
        for path, page in self._pages(built).items():
            assert page.resources, f"{path}: no stylesheet found, so the check saw nothing"
            assert not [u for u in page.resources if re.match(r"(https?:)?//", u)], (path, page.resources)
            assert "default-src 'none'" in page.meta["Content-Security-Policy"], path

    def test_the_markup_describes_the_software_and_invents_no_rating(self, built):
        pages = self._pages(built)
        titles = {page_path(p): front_matter(p)["title"] for p in source_pages()}
        home = [json.loads(b) for b in pages[""].jsonld]
        types = {n["@type"] for n in home[0]["@graph"]}
        assert types == {"WebSite", "SoftwareSourceCode", "Organization"}
        code = next(n for n in home[0]["@graph"] if n["@type"] == "SoftwareSourceCode")
        assert code["version"] == pyproject_version() and code["codeRepository"] == "https://github.com/pctablet505/servedeck"
        for path, page in pages.items():
            blocks = page.jsonld
            assert blocks, path
            assert not re.search(r"aggregateRating|\"review\"|SoftwareApplication", "".join(blocks)), path
            if path:
                crumbs = json.loads(blocks[0])["itemListElement"]
                assert [c["position"] for c in crumbs] == [1, 2], path
                assert crumbs[1]["name"] == titles[path], path

    def test_the_sitemap_lists_each_page_once_at_its_canonical_address(self, built):
        text = (built["__root__"][1] / "sitemap.xml").read_text(encoding="utf-8")
        locs = re.findall(r"<loc>([^<]+)</loc>", text)
        assert sorted(locs) == sorted(SITE_URL + path for path in self._pages(built))

    def test_the_root_files_and_the_not_found_page_ship(self, built):
        root = built["__root__"][1]
        for name in ("robots.txt", "llms.txt", "CNAME"):
            assert (root / name).read_bytes() == (DOCS / name).read_bytes(), name
        missing = _Page((root / "404.html").read_text(encoding="utf-8"))
        assert missing.meta["robots"] == "noindex"
        assert missing.resources == ["/css/site.css"], "the stylesheet must be a root-relative path, never //css/..."
