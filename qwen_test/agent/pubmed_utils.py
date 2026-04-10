"""
PubMed and PMC utility functions for fetching papers and supplementary materials.
"""
import re
import unicodedata
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List
from urllib.parse import urljoin
import torch


def _strip_diacritics(text: str) -> str:
    """Replace accented characters with their ASCII equivalents.

    E.g. Sörme → Sorme, Müller → Muller.
    """
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


class PubMedFetcher:
    """Handles all PubMed and PMC related operations."""

    def __init__(
        self,
        supp_dir: Path,
        ref_dir: Path,
        pdf_dir: Path,
        ncbi_api_key: Optional[str] = None
    ):
        """
        Initialize PubMed fetcher.

        Args:
            supp_dir: Directory to store supplementary files
            ref_dir: Directory to store referenced papers
            pdf_dir: Main PDF directory
            ncbi_api_key: NCBI API key for higher rate limits
        """
        self.supp_dir = supp_dir
        self.ref_dir = ref_dir
        self.pdf_dir = pdf_dir
        self.ncbi_api_key = ncbi_api_key

    def get_pmc_id(self, pmid: str) -> Optional[str]:
        """Convert PMID to PMC ID using NCBI API."""
        try:
            url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
            params = {
                "ids": pmid,
                "format": "json"
            }
            if self.ncbi_api_key:
                params["api_key"] = self.ncbi_api_key

            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                records = data.get("records", [])
                if records and "pmcid" in records[0]:
                    pmc_id = records[0]["pmcid"]
                    print(f"  Found PMC ID: {pmc_id} for PMID {pmid}")
                    return pmc_id
        except Exception as e:
            print(f"  Error getting PMC ID for PMID {pmid}: {e}")
        return None

    def fetch_supplementary_from_pmc(self, pmid: str) -> List[Path]:
        """
        Fetch supplementary materials from PubMed Central.

        Args:
            pmid: PubMed ID

        Returns:
            List of paths to downloaded supplementary files (pdf, docx, xlsx, zip)
        """
        downloaded_files = []

        pmc_id = self.get_pmc_id(pmid)
        if not pmc_id:
            print(f"  No PMC ID found for PMID {pmid}, cannot fetch supplementary")
            return downloaded_files

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        session = requests.Session()
        session.headers.update(headers)

        start_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"

        try:
            print(f"Fetching article page: {start_url}")
            response = session.get(start_url, timeout=30)

            if response.status_code == 200:
                article_url = response.url
                print(f"  Resolved URL: {article_url}")

                supp_pattern = r'href="([^"]+/bin/[^"]+\.(?:pdf|docx|xlsx|zip))"'
                matches = re.findall(supp_pattern, response.text, re.IGNORECASE)
                matches = list(set(matches))

                print(f"  Found {len(matches)} supplementary files.")

                for i, relative_path in enumerate(matches[:5]):
                    file_name = relative_path.split('/')[-1]

                    ext = file_name.rsplit('.', 1)[-1] if '.' in file_name else 'pdf'
                    supp_filename = f"{pmid}_supp_{i + 1}.{ext.lower()}"
                    supp_path = self.supp_dir / supp_filename

                    if not supp_path.exists():
                        # Try PMC first (same site where we found the link), then Europe PMC as fallback
                        supp_urls = [
                            urljoin(article_url, relative_path),
                            urljoin(f'https://europepmc.org/articles/{pmc_id}/bin/', file_name),
                        ]

                        saved = False
                        for supp_url in supp_urls:
                            print(f"  Downloading: {supp_url}")
                            try:
                                download_headers = headers.copy()
                                download_headers['Referer'] = article_url

                                file_response = session.get(supp_url, headers=download_headers, timeout=60, stream=True)
                                content_type = file_response.headers.get('Content-Type', '').lower()

                                if file_response.status_code == 200 and 'html' not in content_type:
                                    with open(supp_path, 'wb') as f:
                                        for chunk in file_response.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                    downloaded_files.append(supp_path)
                                    print(f"    Success! Saved {supp_filename}")
                                    saved = True
                                    break
                                else:
                                    print(f"    Failed (status={file_response.status_code}, type={content_type})")

                            except Exception as e:
                                print(f"    Error: {e}")

                        if not saved:
                            print(f"    Could not download {file_name} from any source")
                    else:
                        print(f"  Skipping {supp_filename} (already exists)")
                        downloaded_files.append(supp_path)
            else:
                print(f"Failed to load article page. Status: {response.status_code}")

        except Exception as e:
            print(f"Script failed: {e}")

        print(f"  Found {len(downloaded_files)} supplementary files")
        return downloaded_files

    def fetch_paper_by_pmid(self, pmid: str) -> Optional[Path]:
        """
        Fetch a paper PDF by PMID from PubMed Central.

        Tries multiple strategies:
        1. Check if PDF already exists locally
        2. Try Europe PMC direct PDF link
        3. Scrape PMC article page for PDF link
        4. Fall back to OA API for direct PDF link

        Args:
            pmid: PubMed ID

        Returns:
            Path to downloaded PDF or None if not available
        """
        main_path = self.pdf_dir / f"{pmid}.pdf"
        ref_path = self.ref_dir / f"{pmid}.pdf"

        if main_path.exists():
            return main_path
        if ref_path.exists():
            return ref_path

        pmc_id = self.get_pmc_id(pmid)
        if not pmc_id:
            return None

        # Extract numeric PMC ID (remove 'PMC' prefix)
        pmc_num = pmc_id.replace('PMC', '')

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        # Strategy 1: Try Europe PMC direct PDF link (most reliable)
        try:
            europepmc_pdf_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmc_id}&blobtype=pdf"
            print(f"  Trying Europe PMC: {europepmc_pdf_url}")
            pdf_response = requests.get(europepmc_pdf_url, headers=headers, timeout=120, allow_redirects=True)
            content_type = pdf_response.headers.get('Content-Type', '')

            if pdf_response.status_code == 200 and ('application/pdf' in content_type or pdf_response.content[:4] == b'%PDF'):
                with open(ref_path, 'wb') as f:
                    f.write(pdf_response.content)
                print(f"  Downloaded PDF from Europe PMC to: {ref_path}")
                return ref_path

        except Exception as e:
            print(f"  Error fetching PDF from Europe PMC: {e}")

        # Strategy 2: Scrape PMC article page for PDF link
        pdf_filename = None
        try:
            article_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
            print(f"  Fetching PMC article page: {article_url}")
            response = requests.get(article_url, headers=headers, timeout=30)

            if response.status_code == 200:
                # Look for PDF links in the page
                pdf_pattern = r'href="(pdf/[^"]+\.pdf)"'
                matches = re.findall(pdf_pattern, response.text)

                if matches:
                    pdf_filename = matches[0].replace('pdf/', '')
                    # Try multiple URL patterns
                    pdf_urls = [
                        f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/{matches[0]}",
                        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/{pdf_filename}",
                    ]

                    for pdf_url in pdf_urls:
                        print(f"  Trying PDF URL: {pdf_url}")
                        pdf_response = requests.get(pdf_url, headers=headers, timeout=120, allow_redirects=True)
                        content_type = pdf_response.headers.get('Content-Type', '')

                        if pdf_response.status_code == 200 and ('application/pdf' in content_type or pdf_response.content[:4] == b'%PDF'):
                            with open(ref_path, 'wb') as f:
                                f.write(pdf_response.content)
                            print(f"  Downloaded PDF to: {ref_path}")
                            return ref_path

        except Exception as e:
            print(f"  Error fetching PDF from PMC page: {e}")

        # Strategy 3: Try OA API (may return tgz or direct PDF)
        try:
            oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmc_id}"
            response = requests.get(oa_url, timeout=30)

            if response.status_code == 200:
                root = ET.fromstring(response.content)

                for record in root.findall(".//record"):
                    for link in record.findall(".//link"):
                        href = link.get("href")
                        format_type = link.get("format")

                        if href and format_type == "pdf":
                            print(f"  Downloading reference paper PMID {pmid} from OA API...")
                            pdf_response = requests.get(href, headers=headers, timeout=120, allow_redirects=True)
                            if pdf_response.status_code == 200 and pdf_response.content[:4] == b'%PDF':
                                with open(ref_path, 'wb') as f:
                                    f.write(pdf_response.content)
                                print(f"  Saved to: {ref_path}")
                                return ref_path

        except Exception as e:
            print(f"  Error fetching paper from OA API for PMID {pmid}: {e}")

        return None

    def get_pdf_path(self, pmid: str) -> Optional[Path]:
        """Get the path to a PDF by PMID, checking multiple directories."""
        main_path = self.pdf_dir / f"{pmid}.pdf"
        if main_path.exists():
            return main_path

        ref_path = self.ref_dir / f"{pmid}.pdf"
        if ref_path.exists():
            return ref_path

        return None


class CitationSearcher:
    """Handles citation-based PMID searching."""

    def __init__(self, ncbi_api_key: Optional[str] = None, model=None, processor=None, device: str = "cuda"):
        """
        Initialize citation searcher.

        Args:
            ncbi_api_key: NCBI API key for higher rate limits
            model: Qwen3-VL model for title extraction (optional)
            processor: Qwen3-VL processor for title extraction (optional)
            device: Device for model inference
        """
        self.ncbi_api_key = ncbi_api_key
        self.model = model
        self.processor = processor
        self.device = device

    def search_pmid_by_citation(self, citation: str) -> Optional[str]:
        """
        Search for a PMID using a citation string.

        Tries multiple strategies:
        1. Split compound references like "(a) ... ; (b) ..." and search each
        2. Extract and search by DOI
        3. Extract and search by title
        4. Free text search using author names, journal, and year

        Args:
            citation: Citation string like "Collie, G. W.; Koh, C. M.; ... ACS Med. Chem. Lett. 2019, 10, 1322-1327."

        Returns:
            PMID string if found, None otherwise
        """
        print(f"  Searching PMID for citation: {citation[:80]}...")

        # Split compound references: "(a) Author... ; (b) Author..."
        sub_parts = re.split(r';\s*\([a-z]\)\s*', citation)
        if len(sub_parts) > 1:
            # Strip leading "(a) " from first part
            sub_parts[0] = re.sub(r'^\s*\([a-z]\)\s*', '', sub_parts[0])
            for i, part in enumerate(sub_parts):
                part = part.strip()
                if not part:
                    continue
                print(f"    Searching sub-reference ({chr(ord('a') + i)}): {part[:60]}...")
                pmid = self._search_single_citation(part)
                if pmid:
                    return pmid
            print(f"    Could not find PMID for any sub-reference")
            return None

        return self._search_single_citation(citation)

    def _search_single_citation(self, citation: str) -> Optional[str]:
        """Search for a PMID from a single (non-compound) citation string."""

        # Strategy 1: Try to find DOI in the citation
        doi_pattern = r'10\.\d{4,}/[^\s,;]+'
        doi_match = re.search(doi_pattern, citation)
        if doi_match:
            doi = doi_match.group(0).rstrip('.')
            print(f"    Found DOI: {doi}")
            pmid = self._search_pubmed_by_doi(doi)
            if pmid:
                return pmid

        # Strategy 2: Try to extract and search by title
        title = self._extract_title_from_citation(citation)
        year_match = re.search(r'\b(19|20)\d{2}\b', citation)
        year = year_match.group(0) if year_match else None
        if title:
            print(f"    Extracted title: {title[:60]}...")
            pmid = self._search_pubmed_by_title(title, year=year)
            if pmid:
                return pmid

        # Strategy 3: Free text search using parts of the citation
        pmid = self._search_pubmed_by_text(citation)
        if pmid:
            return pmid

        print(f"    Could not find PMID for citation")
        return None

    def _search_pubmed_by_doi(self, doi: str) -> Optional[str]:
        """Search PubMed by DOI."""
        try:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": f"{doi}[doi]",
                "retmode": "json",
                "retmax": 1
            }
            if self.ncbi_api_key:
                params["api_key"] = self.ncbi_api_key

            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                id_list = data.get("esearchresult", {}).get("idlist", [])
                if id_list:
                    pmid = id_list[0]
                    print(f"    Found PMID by DOI: {pmid}")
                    return pmid
        except Exception as e:
            print(f"    Error searching by DOI: {e}")
        return None

    def _extract_title_from_citation(self, citation: str) -> Optional[str]:
        """
        Extract the title from a citation string.

        Uses Qwen3-VL model if available, otherwise falls back to regex-based extraction.

        Args:
            citation: Citation string like "Author, A.; Author, B. Title of the Paper. Journal Name Year, Vol, Pages."

        Returns:
            Extracted title or None if extraction failed
        """
        # Try model-based extraction if model is available
        if self.model is not None and self.processor is not None:
            title = self._extract_title_with_model(citation)
            if title:
                return title

        # Fallback to regex-based extraction
        return self._extract_title_with_regex(citation)

    def _extract_title_with_model(self, citation: str) -> Optional[str]:
        """
        Use Qwen3-VL model to extract the title from a citation string.

        Args:
            citation: Citation string

        Returns:
            Extracted title or None if extraction failed
        """
        prompt = f"""Extract ONLY the paper title from this citation. Return just the title text, nothing else.

        Citation: {citation}

        Title:"""

        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text_prompt],
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            title = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            # Clean up the title
            title = title.strip('"\'')
            # Remove any trailing punctuation that might be from journal name
            title = re.sub(r'\.\s*$', '', title)

            if title and len(title) > 10:
                print(f"    [Model] Extracted title: {title[:60]}...")
                return title

        except Exception as e:
            print(f"    Error extracting title with model: {e}")

        return None

    def _extract_title_with_regex(self, citation: str) -> Optional[str]:
        """
        Extract the title from a citation string using regex patterns.

        Common formats:
        - "Author, A.; Author, B. Title of the Paper. Journal Name Year, Vol, Pages."

        Args:
            citation: Citation string

        Returns:
            Extracted title or None if extraction failed
        """
        journal_patterns = [
            r'\.\s+[A-Z][a-z]*\.\s+[A-Z][a-z]*\.\s+(?:Chem|Med|Biol|Biochem)',
            r'\.\s+[A-Z][A-Z]+\s+[A-Z][a-z]+',
            r'\.\s+Nature\b',
            r'\.\s+Science\b',
            r'\.\s+Cell\b',
            r'\.\s+PNAS\b',
            r'\.\s+Proc\.\s+Natl',
            r'\.\s+\d{4}\s*[,;]',
        ]

        for journal_pattern in journal_patterns:
            match = re.search(journal_pattern, citation)
            if match:
                text_before = citation[:match.start()]
                sentences = text_before.split('. ')
                if len(sentences) >= 2:
                    title = sentences[-1].strip()
                    if len(title) > 10:
                        return title

        # Fallback: Look for capitalized phrase that looks like a title
        title_pattern = r'[A-Z][a-z]+(?:\s+[A-Za-z]+){3,}'
        matches = re.findall(title_pattern, citation)
        for match in matches:
            if len(match) > 20 and ',' not in match:
                return match

        return None

    def _search_pubmed_by_title(self, title: str, year: Optional[str] = None) -> Optional[str]:
        """Search PubMed by title, optionally filtered by publication year."""
        try:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            clean_title = re.sub(r'[^\w\s]', ' ', title)
            term = f"{clean_title}[Title]"
            if year:
                term += f" AND {year}[Date - Publication]"
            params = {
                "db": "pubmed",
                "term": term,
                "retmode": "json",
                "retmax": 1
            }
            if self.ncbi_api_key:
                params["api_key"] = self.ncbi_api_key

            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                id_list = data.get("esearchresult", {}).get("idlist", [])
                if id_list:
                    pmid = id_list[0]
                    print(f"    Found PMID by title: {pmid}")
                    return pmid
        except Exception as e:
            print(f"    Error searching by title: {e}")
        return None

    def _search_pubmed_by_text(self, citation: str) -> Optional[str]:
        """Free text search using author names, journal, year, and optionally volume/page."""
        try:
            search_terms = []

            # Extract first author's last name (strip leading (a)/(b)/number prefixes)
            clean_citation = re.sub(r'^\s*(?:\([a-z]\)\s*|\d+\.\s*)', '', citation)
            author_match = re.match(r'^([A-Za-z\u00C0-\u024F]+)', clean_citation)
            if author_match:
                author = _strip_diacritics(author_match.group(1))
                search_terms.append(f"{author}[Author]")

            # Extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', citation)
            if year_match:
                search_terms.append(f"{year_match.group(0)}[Date - Publication]")

            # Extract journal abbreviation
            # Dots after abbreviated words are optional so both
            # "J. Med. Chem." and "J Med Chem." are matched.
            def _opt_dot(abbr: str) -> str:
                """Make dots optional in a journal abbreviation pattern.

                E.g. 'J. Med. Chem.' becomes a pattern matching both
                'J. Med. Chem.' and 'J Med Chem.' (dots optional).
                """
                parts = abbr.replace('.', '').split()
                return r'\s+'.join(re.escape(p) + r'\.?' for p in parts)

            journal_patterns = [
                (_opt_dot('J. Med. Chem.'), "J Med Chem"),
                (_opt_dot('ACS Med. Chem. Lett.'), "ACS Med Chem Lett"),
                (_opt_dot('ACS Chem. Biol.'), "ACS Chem Biol"),
                (_opt_dot('Bioorg. Med. Chem. Lett.'), "Bioorg Med Chem Lett"),
                (_opt_dot('Bioorg. Med. Chem.'), "Bioorg Med Chem"),
                (_opt_dot('Eur. J. Med. Chem.'), "Eur J Med Chem"),
                (r'ChemMedChem', "ChemMedChem"),
                (r'ChemBioChem', "ChemBioChem"),
                (_opt_dot('Cell Chem. Biol.'), "Cell Chem Biol"),
                (r'(?<!\w)Chem\.?\s*Biol\.?(?!\s*Lett)', "Chemistry & biology"),
                (_opt_dot('J. Biol. Chem.'), "J Biol Chem"),
                (r'Biochemistry', "Biochemistry"),
                (_opt_dot('Angew. Chem.'), "Angew Chem Int Ed Engl"),
                (_opt_dot('Proc. Natl. Acad. Sci.'), "Proc Natl Acad Sci U S A"),
                (r'\bPNAS\b', "Proc Natl Acad Sci U S A"),
                (_opt_dot('J. Am. Chem. Soc.'), "J Am Chem Soc"),
                (_opt_dot('Nat. Chem. Biol.'), "Nat Chem Biol"),
                (_opt_dot('Nat. Struct. Mol. Biol.'), "Nat Struct Mol Biol"),
                (r'\bNature\b', "Nature"),
                (r'\bScience\b', "Science"),
                # Additional journals
                (_opt_dot('Org. Biomol. Chem.'), "Org Biomol Chem"),
                (_opt_dot('Anal. Biochem.'), "Anal Biochem"),
                (_opt_dot('Meth. Enzymol.'), "Methods Enzymol"),
                (_opt_dot('Methods Enzymol.'), "Methods Enzymol"),
                (_opt_dot('J. Biomol. Screening'), "J Biomol Screen"),
                (_opt_dot('J. Biomol. Screen.'), "J Biomol Screen"),
                (_opt_dot('Chem. Eur. J.'), "Chemistry"),
                (_opt_dot('J. Org. Chem.'), "J Org Chem"),
                (_opt_dot('Org. Lett.'), "Org Lett"),
                (_opt_dot('Mol. Pharmacol.'), "Mol Pharmacol"),
                (_opt_dot('Cancer Res.'), "Cancer Res"),
                (_opt_dot('Clin. Cancer Res.'), "Clin Cancer Res"),
            ]
            journal_name = None
            for pattern, jname in journal_patterns:
                if re.search(pattern, citation, re.IGNORECASE):
                    search_terms.append(f'"{jname}"[Journal]')
                    journal_name = jname
                    break

            # Extract volume and start page for narrowing
            # Supports: "2005, 3, 1922" and "2014;57:567" and "2014; 57: 567"
            volume_match = re.search(r'(?:19|20)\d{2}[,;]\s*(\d+)[,;:]\s*\d+', citation)
            page_match = re.search(r'(?:19|20)\d{2}[,;]\s*\d+[,;:]\s*(\d+)', citation)

            def _run_search(terms, max_results=5):
                url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                query = " AND ".join(terms)
                params = {
                    "db": "pubmed",
                    "term": query,
                    "retmode": "json",
                    "retmax": max_results
                }
                if self.ncbi_api_key:
                    params["api_key"] = self.ncbi_api_key
                print(f"    Free text search: {query}")
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json().get("esearchresult", {}).get("idlist", [])
                return []

            def _resolve_ids(id_list, base_terms):
                if not id_list:
                    return None
                if len(id_list) == 1:
                    return id_list[0]
                # Narrow with volume and/or page
                narrow_terms = list(base_terms)
                if volume_match:
                    narrow_terms.append(f"{volume_match.group(1)}[Volume]")
                if page_match:
                    narrow_terms.append(f"{page_match.group(1)}[Page]")
                if len(narrow_terms) > len(base_terms):
                    narrow_ids = _run_search(narrow_terms, max_results=1)
                    if narrow_ids:
                        print(f"    Found PMID by narrowed search: {narrow_ids[0]}")
                        return narrow_ids[0]
                # Fall back to first broad result
                return id_list[0]

            if len(search_terms) >= 2:
                id_list = _run_search(search_terms)
                pmid = _resolve_ids(id_list, search_terms)
                if pmid:
                    print(f"    Found PMID by free text: {pmid}")
                    return pmid

            # Fallback: search by year + journal + volume + page only (skips author,
            # handles compound surnames like "VanMolle" that don't match PubMed index)
            if journal_name and year_match and (volume_match or page_match):
                fallback_terms = [
                    f'"{journal_name}"[Journal]',
                    f"{year_match.group(0)}[Date - Publication]",
                ]
                if volume_match:
                    fallback_terms.append(f"{volume_match.group(1)}[Volume]")
                if page_match:
                    fallback_terms.append(f"{page_match.group(1)}[Page]")
                fb_ids = _run_search(fallback_terms, max_results=1)
                if fb_ids:
                    print(f"    Found PMID by journal/year/vol/page fallback: {fb_ids[0]}")
                    return fb_ids[0]
        except Exception as e:
            print(f"    Error in free text search: {e}")
        return None

    def extract_referenced_pmids(self, text: str) -> List[str]:
        """
        Extract PMIDs from text that references previous literature.

        Args:
            text: Text to search for references

        Returns:
            List of PMIDs mentioned in the text
        """
        pmids = []

        # Pattern for direct PMID mentions
        pmid_pattern = r'PMID[:\s]*(\d{7,8})'
        pmids.extend(re.findall(pmid_pattern, text, re.IGNORECASE))

        # Pattern for PubMed URLs
        pubmed_url_pattern = r'pubmed\.ncbi\.nlm\.nih\.gov/(\d{7,8})'
        pmids.extend(re.findall(pubmed_url_pattern, text))

        return list(set(pmids))

    def extract_citations_from_text(self, text: str) -> List[str]:
        """
        Extract citation strings from text.

        Args:
            text: Text that may contain citations

        Returns:
            List of extracted citation strings
        """
        citations = []

        # Pattern for numbered reference lists like "(26) Author... (27) Author..."
        # Split on reference number markers and extract each individual citation
        numbered_ref_pattern = r'\(\d+\)\s+([A-Z].*?)(?=\s*\(\d+\)\s+[A-Z]|$)'
        numbered_matches = re.findall(numbered_ref_pattern, text, re.DOTALL)
        if numbered_matches:
            for match in numbered_matches:
                match = match.strip()
                # Only include if it looks like a proper citation (has a year)
                if re.search(r'(?:19|20)\d{2}', match) and len(match) > 20:
                    citations.append(match)
            return citations

        # Pattern for typical citation format
        citation_pattern = r'[A-Z][a-z]+,\s*[A-Z]\.(?:\s*[A-Z]\.)?(?:;\s*[A-Z][a-z]+,\s*[A-Z]\.(?:\s*[A-Z]\.)?)*[^.]*(?:19|20)\d{2}[^.]*\d+[-–]\d+\.?'
        matches = re.findall(citation_pattern, text)
        citations.extend(matches)

        # Pattern for parenthetical references
        paren_pattern = r'\(([^)]*(?:19|20)\d{2}[^)]*)\)'
        paren_matches = re.findall(paren_pattern, text)
        for match in paren_matches:
            if re.search(r'[A-Z][a-z]+.*(?:19|20)\d{2}', match):
                citations.append(match)

        return list(set(citations))
