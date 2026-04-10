"""
Unit tests for reference-chasing logic in BaseAssayExtractionAgent.

Tests the new methods without requiring GPU or model loading by mocking
the model-dependent parts.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).parent))


class FakeBaseAgent:
    """Minimal stub that inherits the real base class methods without loading a model."""

    def __init__(self):
        self.search_references = True
        self.max_reference_depth = 1
        self.citation_searcher = MagicMock()
        # Default: no PMIDs found from citations
        self.citation_searcher.extract_referenced_pmids.return_value = []
        self.citation_searcher.extract_citations_from_text.return_value = []
        self.citation_searcher.search_pmid_by_citation.return_value = None

    # Bind the real methods from the base class
    from agent.base_extraction_agent import BaseAssayExtractionAgent as _Base
    _is_paragraph_found = _Base._is_paragraph_found
    _combine_paragraphs = _Base._combine_paragraphs
    _resolve_reference_pmids = _Base._resolve_reference_pmids
    _check_and_fetch_references = _Base._check_and_fetch_references
    _search_references_not_found = _Base._search_references_not_found

    def _extract_for_reference(self, pmid, assay_description, max_pages, max_new_tokens, _depth):
        """Will be mocked per test."""
        raise NotImplementedError


class TestCombineParagraphs(unittest.TestCase):
    """Test _combine_paragraphs with different input types."""

    def test_both_dict(self):
        main = {"Methods": "main text"}
        ref = {"Results": "ref text"}
        combined = FakeBaseAgent._combine_paragraphs(main, "111", ref, "222")
        self.assertIn("[PMID 111] Methods", combined)
        self.assertIn("[Ref PMID 222] Results", combined)
        self.assertEqual(combined["[PMID 111] Methods"], "main text")
        self.assertEqual(combined["[Ref PMID 222] Results"], "ref text")

    def test_both_string(self):
        combined = FakeBaseAgent._combine_paragraphs("main text", "111", "ref text", "222")
        self.assertIn("[From PMID 111]", combined)
        self.assertIn("[From referenced PMID 222]", combined)

    def test_mixed_dict_string(self):
        combined = FakeBaseAgent._combine_paragraphs(
            {"Methods": "main"}, "111", "ref text", "222"
        )
        self.assertIsInstance(combined, dict)
        self.assertIn("[PMID 111] Methods", combined)
        self.assertIn("[Ref PMID 222]", combined)


class TestResolveReferencePmids(unittest.TestCase):
    """Test _resolve_reference_pmids with direct and citation-based fallback."""

    def test_direct_pmid_found(self):
        agent = FakeBaseAgent()
        agent.citation_searcher.extract_referenced_pmids.return_value = ["99999"]
        result = agent._resolve_reference_pmids("see PMID 99999", {}, "11111")
        self.assertEqual(result, ["99999"])

    def test_excludes_current_pmid(self):
        agent = FakeBaseAgent()
        agent.citation_searcher.extract_referenced_pmids.return_value = ["11111", "22222"]
        result = agent._resolve_reference_pmids("refs", {}, "11111")
        self.assertEqual(result, ["22222"])

    def test_citation_fallback(self):
        agent = FakeBaseAgent()
        agent.citation_searcher.extract_referenced_pmids.return_value = []
        agent.citation_searcher.extract_citations_from_text.return_value = ["Smith et al. 2020"]
        agent.citation_searcher.search_pmid_by_citation.return_value = "33333"
        result = agent._resolve_reference_pmids(
            "as described by Smith et al. 2020", {}, "11111"
        )
        self.assertIn("33333", result)

    def test_no_pmids_found(self):
        agent = FakeBaseAgent()
        result = agent._resolve_reference_pmids("some vague reference", {}, "11111")
        self.assertEqual(result, [])


class TestCheckAndFetchReferences(unittest.TestCase):
    """Test _check_and_fetch_references (post-found reference chasing)."""

    def test_no_references_returns_unchanged(self):
        agent = FakeBaseAgent()
        result = {
            "original_paragraph": {"Methods": "found text"},
            "references_previous": "none",
            "search_path": ["main"],
            "supplementary_source": [],
        }
        out = agent._check_and_fetch_references(result, "111", "desc", None, 2048, 0)
        self.assertEqual(out["original_paragraph"], {"Methods": "found text"})

    def test_disabled_returns_unchanged(self):
        agent = FakeBaseAgent()
        agent.search_references = False
        result = {
            "original_paragraph": {"Methods": "found text"},
            "references_previous": "see PMID 99999",
            "search_path": ["main"],
            "supplementary_source": [],
        }
        out = agent._check_and_fetch_references(result, "111", "desc", None, 2048, 0)
        self.assertEqual(out["original_paragraph"], {"Methods": "found text"})

    def test_depth_exceeded_returns_unchanged(self):
        agent = FakeBaseAgent()
        result = {
            "original_paragraph": {"Methods": "found text"},
            "references_previous": "see PMID 99999",
            "search_path": ["main"],
            "supplementary_source": [],
        }
        out = agent._check_and_fetch_references(result, "111", "desc", None, 2048, 1)
        self.assertEqual(out["original_paragraph"], {"Methods": "found text"})

    def test_combines_when_reference_found(self):
        agent = FakeBaseAgent()
        agent.citation_searcher.extract_referenced_pmids.return_value = ["99999"]

        ref_result = {
            "original_paragraph": {"Protocol": "reference protocol details"},
            "search_path": ["main"],
            "supplementary_source": [],
        }
        agent._extract_for_reference = MagicMock(return_value=ref_result)

        result = {
            "original_paragraph": {"Methods": "briefly describes assay"},
            "references_previous": "see PMID 99999 for full protocol",
            "source": "main_paper_111",
            "search_path": ["main"],
            "supplementary_source": [],
        }
        out = agent._check_and_fetch_references(result, "111", "desc", None, 2048, 0)

        # Should be combined
        self.assertIn("[PMID 111] Methods", out["original_paragraph"])
        self.assertIn("[Ref PMID 99999] Protocol", out["original_paragraph"])
        self.assertEqual(out["search_path"], ["main", "reference", "main"])
        self.assertIn("referenced_paper_99999", out["source"])

    def test_no_combine_when_reference_not_found(self):
        agent = FakeBaseAgent()
        agent.citation_searcher.extract_referenced_pmids.return_value = ["99999"]

        ref_result = {
            "original_paragraph": {},
            "search_path": [],
            "supplementary_source": [],
        }
        agent._extract_for_reference = MagicMock(return_value=ref_result)

        result = {
            "original_paragraph": {"Methods": "briefly describes assay"},
            "references_previous": "see PMID 99999",
            "source": "main_paper_111",
            "search_path": ["main"],
            "supplementary_source": [],
        }
        out = agent._check_and_fetch_references(result, "111", "desc", None, 2048, 0)

        # Should be unchanged
        self.assertEqual(out["original_paragraph"], {"Methods": "briefly describes assay"})


class TestSearchReferencesNotFound(unittest.TestCase):
    """Test _search_references_not_found (not-found reference search)."""

    def test_no_references_returns_unchanged(self):
        agent = FakeBaseAgent()
        result = {
            "original_paragraph": {},
            "references_previous": "none",
            "search_path": [],
            "supplementary_source": [],
        }
        out = agent._search_references_not_found(result, "111", "desc", None, 2048, 0)
        self.assertEqual(out["original_paragraph"], {})

    def test_returns_reference_result_when_found(self):
        agent = FakeBaseAgent()
        agent.citation_searcher.extract_referenced_pmids.return_value = ["99999"]

        ref_result = {
            "original_paragraph": {"Protocol": "full protocol from ref paper"},
            "search_path": ["main"],
            "supplementary_source": [],
            "source": "main_paper_99999",
        }
        agent._extract_for_reference = MagicMock(return_value=ref_result)

        result = {
            "original_paragraph": {},
            "references_previous": "as described in PMID 99999",
            "search_path": [],
            "supplementary_source": [],
        }
        out = agent._search_references_not_found(result, "111", "desc", None, 2048, 0)

        self.assertIn("Protocol", str(out["original_paragraph"]))
        self.assertEqual(out["search_path"], ["reference", "main"])
        self.assertIn("referenced_paper_99999_from_111", out["source"])

    def test_tries_multiple_pmids(self):
        agent = FakeBaseAgent()
        agent.citation_searcher.extract_referenced_pmids.return_value = ["88888", "99999"]

        # First ref not found, second found
        ref_results = [
            {"original_paragraph": {}, "search_path": [], "supplementary_source": []},
            {
                "original_paragraph": {"Methods": "found in second ref"},
                "search_path": ["main"],
                "supplementary_source": [],
                "source": "main_paper_99999",
            },
        ]
        agent._extract_for_reference = MagicMock(side_effect=ref_results)

        result = {
            "original_paragraph": {},
            "references_previous": "see refs 88888 and 99999",
            "search_path": [],
            "supplementary_source": [],
        }
        out = agent._search_references_not_found(result, "111", "desc", None, 2048, 0)

        self.assertTrue(agent._is_paragraph_found(out["original_paragraph"]))
        self.assertEqual(agent._extract_for_reference.call_count, 2)


class TestIsParagraphFound(unittest.TestCase):
    """Test _is_paragraph_found with various input types."""

    def test_none(self):
        self.assertFalse(FakeBaseAgent._is_paragraph_found(None))

    def test_empty_dict(self):
        self.assertFalse(FakeBaseAgent._is_paragraph_found({}))

    def test_dict_with_content(self):
        self.assertTrue(FakeBaseAgent._is_paragraph_found({"Methods": "some text"}))

    def test_dict_with_not_found(self):
        self.assertFalse(FakeBaseAgent._is_paragraph_found({"Methods": "NOT FOUND"}))

    def test_string_not_found(self):
        self.assertFalse(FakeBaseAgent._is_paragraph_found("NOT FOUND"))

    def test_string_with_content(self):
        self.assertTrue(FakeBaseAgent._is_paragraph_found("some real paragraph"))

    def test_error_string(self):
        self.assertFalse(FakeBaseAgent._is_paragraph_found("ERROR: something went wrong"))


if __name__ == "__main__":
    unittest.main()
