from pathlib import Path
import subprocess
import tempfile
import unittest

from check_repository import audit


class RepositoryHygieneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.git("init", "-q")
        self.write(".gitignore", "build/\n__pycache__/\nresults/**/*.png\n")
        self.write("README.md", "# Project\n")
        self.git("add", "--", ".gitignore", "README.md")

    def git(self, *arguments):
        return subprocess.check_output(["git", "-C", str(self.root), *arguments])

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def test_links_fragments_duplicates_unicode_and_encoded_spaces(self):
        self.write("docs/a b.md", "# Café & tools\n## Repeat\n## Repeat\n## `Code`\n")
        self.write("README.md", "# Project\n[Unicode](docs/a%20b.md#café--tools)\n"
                   "[Duplicate](<docs/a b.md#repeat-1>)\n[Code](docs/a%20b.md#code)\n"
                   "[Folder](docs/)\n[Root](/docs/a%20b.md#repeat)\n")
        self.assertEqual(audit(self.root), ([], 2, 5))

    def test_reference_links_and_inline_titles(self):
        self.write("README.md", '# Project\n[First][guide]\n[guide][]\n[guide]\n'
                   '[Inline](README.md#project "Title")\n\n[guide]: README.md#project\n')
        self.assertEqual(audit(self.root), ([], 1, 4))

    def test_examples_and_external_urls_are_not_probed(self):
        self.write("README.md", '# Project\n```markdown\n[Example](missing.md)\n# Fake heading\n```\n'
                   '`[Code](missing.md)`\n~~~\n[Example](missing.md)\n~~~\n'
                   '[Web](https://example.invalid/missing)\n[Email](mailto:example@example.invalid)\n')
        self.assertEqual(audit(self.root), ([], 1, 0))

    def test_missing_targets_and_headings_have_source_lines(self):
        self.write("README.md", "# Project\n[File](absent.md)\n[Heading](README.md#absent)\n")
        errors, _, links = audit(self.root)
        self.assertEqual(links, 2)
        self.assertTrue(any("README.md:2: missing target" in error for error in errors))
        self.assertTrue(any("README.md:3: missing heading" in error for error in errors))

    def test_worktree_deletions_and_new_documents_are_checked_without_staging(self):
        self.write("old.md", "# Old\n")
        self.git("add", "--", "old.md")
        (self.root / "old.md").unlink()
        self.write("docs/new.md", "[Broken](missing.md)\n")
        before = self.git("diff", "--cached", "--name-status")
        errors, documents, _ = audit(self.root)
        self.assertEqual(documents, 2)
        self.assertEqual(len(errors), 1)
        self.assertIn("docs/new.md:1", errors[0])
        self.assertEqual(before, self.git("diff", "--cached", "--name-status"))

    def test_tracked_artifacts_fail_but_untracked_caches_and_result_csvs_are_allowed(self):
        self.write("__pycache__/local.pyc", "cache")
        self.write("results/measurements.csv", "paths,price\n1000,10\n")
        self.git("add", "--", "results/measurements.csv")
        self.assertEqual(audit(self.root)[0], [])
        for path in ("build/program.o", "__pycache__/forced.pyc", "results/plot.png",
                     "scratch/CMakeCache.txt", "debug.log"):
            with self.subTest(path=path):
                self.write(path, "artifact")
                self.git("add", "-f", "--", path)
                errors, _, _ = audit(self.root)
                self.assertTrue(any(path in error and "tracked" in error for error in errors))
                (self.root / path).unlink()
        self.assertEqual(audit(self.root)[0], [])

    def test_links_cannot_escape_the_checkout(self):
        self.write("README.md", "# Project\n[Outside](../outside.md)\n")
        self.assertIn("link escapes repository", audit(self.root)[0][0])


if __name__ == "__main__":
    unittest.main()
