import base64
import json
import os
import re
import tempfile
import uuid
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models


def delete_existing_source_space(
    qdrant_host: str,
    source_type: str,
    space_key: str,
    *,
    keep_ingest_run_id: str | None = None,
    only_ingest_run_id: str | None = None,
):
    conditions = [
        models.FieldCondition(
            key="metadata.source_type",
            match=models.MatchValue(value=source_type),
        ),
        models.FieldCondition(
            key="metadata.space",
            match=models.MatchValue(value=space_key),
        ),
    ]
    if only_ingest_run_id:
        conditions.append(
            models.FieldCondition(
                key="metadata.ingest_run_id",
                match=models.MatchValue(value=only_ingest_run_id),
            )
        )

    client = QdrantClient(url=qdrant_host)
    client.delete(
        collection_name="confluence_docs",
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=conditions,
                must_not=[
                    models.FieldCondition(
                        key="metadata.ingest_run_id",
                        match=models.MatchValue(value=keep_ingest_run_id),
                    )
                ]
                if keep_ingest_run_id
                else None,
            )
        ),
        wait=True,
    )


def ingest_label(source_type: str, source_key: str, source_name: str | None = None):
    name_part = f"[{source_name}]" if source_name and source_name != source_key else ""
    return f"[INGEST][{source_type}:{source_key}]{name_part}"


def save_source_documents(
    qdrant_host: str,
    source_type: str,
    space_key: str,
    documents: list[Document],
    embeddings: OllamaEmbeddings,
    replace_existing: bool,
):
    ingest_run_id = uuid.uuid4().hex
    for document in documents:
        document.metadata["ingest_run_id"] = ingest_run_id

    try:
        QdrantVectorStore.from_documents(
            documents,
            embeddings,
            url=qdrant_host,
            collection_name="confluence_docs",
            force_recreate=False,
        )
    except Exception:
        try:
            delete_existing_source_space(
                qdrant_host,
                source_type,
                space_key,
                only_ingest_run_id=ingest_run_id,
            )
        except Exception as cleanup_error:
            print(f"{ingest_label(source_type, space_key)}[rollback-failed] error={cleanup_error}")
        raise

    if replace_existing:
        print(f"{ingest_label(source_type, space_key)}[delete-stale] keep_run={ingest_run_id}")
        delete_existing_source_space(
            qdrant_host,
            source_type,
            space_key,
            keep_ingest_run_id=ingest_run_id,
        )

    return ingest_run_id


def normalize_atlassian_site_url(url: str) -> str:
    normalized = (url or "").rstrip("/")
    for suffix in ("/wiki", "/jira"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def normalize_confluence_url(url: str) -> str:
    site_url = normalize_atlassian_site_url(url)
    if "atlassian.net" in site_url:
        return f"{site_url}/wiki"
    return site_url


def jira_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(filter(None, [jira_value_to_text(item) for item in value]))
    if isinstance(value, dict):
        for key in ("value", "name", "displayName", "emailAddress"):
            if value.get(key):
                return jira_value_to_text(value.get(key))
        if value.get("content"):
            return jira_value_to_text(value.get("content"))
        return ", ".join(
            f"{key}: {text}"
            for key, item in value.items()
            if key not in {"self", "avatarUrls", "iconUrl"} and (text := jira_value_to_text(item))
        )
    return str(value).strip()


def format_jira_custom_fields(fields: dict[str, Any], field_names: dict[str, str]) -> list[str]:
    lines = []
    for field_id, value in sorted(fields.items()):
        if not field_id.startswith("customfield_"):
            continue
        text = jira_value_to_text(value)
        if not text or text in {"{}", "[]"}:
            continue
        field_name = field_names.get(field_id, field_id)
        lines.append(f"{field_name}: {text[:500]}")
    return lines


def jira_custom_field_map(fields: dict[str, Any], field_names: dict[str, str]) -> dict[str, str]:
    values = {}
    for field_id, value in sorted(fields.items()):
        if not field_id.startswith("customfield_"):
            continue
        text = jira_value_to_text(value)
        if not text or text in {"{}", "[]"}:
            continue
        values[field_names.get(field_id, field_id)] = text[:500]
    return values


def jira_date_fields(custom_fields: dict[str, str]) -> dict[str, str]:
    date_fields = {}
    date_name_tokens = ("date", "날짜", "일자", "일시", "점검일")
    date_value_pattern = re.compile(r"\b20\d{2}[-./]\d{1,2}[-./]\d{1,2}\b|\b\d{1,2}월\s*\d{1,2}일\b")
    for name, value in custom_fields.items():
        normalized_name = name.lower()
        if any(token in normalized_name for token in date_name_tokens) or date_value_pattern.search(value):
            date_fields[name] = value
    return date_fields


def normalize_jira_status_category(status: dict[str, Any]) -> str:
    category = status.get("statusCategory") or {}
    category_key = (category.get("key") or "").lower()
    if category_key == "new":
        return "todo"
    if category_key == "indeterminate":
        return "in_progress"
    if category_key == "done":
        return "done"

    status_name = (status.get("name") or "").lower()
    if any(token in status_name for token in ("done", "closed", "resolved", "완료")):
        return "done"
    if any(token in status_name for token in ("progress", "진행")):
        return "in_progress"
    return "todo"


class ConfluenceIngestor:
    JIRA_ISSUE_KEY_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")

    def __init__(self):
        base_url = normalize_confluence_url(os.getenv("CONFLUENCE_URL", ""))
        self.url = base_url
        self.site_url = normalize_atlassian_site_url(base_url)
        self.jira_url = normalize_atlassian_site_url(os.getenv("JIRA_URL") or self.site_url)
        self.email = os.getenv("CONFLUENCE_EMAIL")
        self.api_token = os.getenv("CONFLUENCE_API_TOKEN")
        self.jira_email = os.getenv("JIRA_EMAIL", self.email)
        self.jira_api_token = os.getenv("JIRA_API_TOKEN", self.api_token)
        self.pat = os.getenv("CONFLUENCE_PAT")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
        self.request_timeout = float(os.getenv("CONFLUENCE_REQUEST_TIMEOUT_SECONDS", "90"))
        self.max_chunk_chars = int(os.getenv("INGEST_MAX_CHUNK_CHARS", "6000"))
        self.last_source_name = None
        self.jira_issue_cache = {}
        self.jira_project_cache = {}
        self.jira_field_names = None

        self._converter = None
        self._converter_unavailable = False
        self.embeddings = OllamaEmbeddings(base_url=self.ollama_host, model=self.embedding_model)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def converter(self):
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter

                self._converter = DocumentConverter()
            except Exception:
                self._converter_unavailable = True
                raise
        return self._converter

    def get_auth_header(self):
        headers = {"Accept": "application/json"}
        if self.email and self.api_token:
            credentials = f"{self.email}:{self.api_token}"
            encoded_auth = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_auth}"
        elif self.pat and ":" in self.pat:
            encoded_auth = base64.b64encode(self.pat.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_auth}"
        elif self.pat:
            headers["Authorization"] = f"Bearer {self.pat}"
        return headers

    def get_jira_auth_header(self):
        headers = {"Accept": "application/json"}
        if self.jira_email and self.jira_api_token:
            credentials = f"{self.jira_email}:{self.jira_api_token}"
            encoded_auth = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_auth}"
        else:
            headers.update(self.get_auth_header())
        return headers

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.get(url, headers=self.get_auth_header(), params=params, timeout=self.request_timeout)
        if response.status_code >= 400:
            detail = response.text[:500]
            raise RuntimeError(f"Confluence API error {response.status_code} for {response.url}: {detail}")
        return response.json()

    def _get_jira_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.get(
            f"{self.jira_url}{path}",
            headers=self.get_jira_auth_header(),
            params=params,
            timeout=self.request_timeout,
        )
        if response.status_code >= 400:
            detail = response.text[:500]
            raise RuntimeError(f"Jira API error {response.status_code} for {response.url}: {detail}")
        return response.json()

    def fetch_jira_field_names(self) -> dict[str, str]:
        if self.jira_field_names is not None:
            return self.jira_field_names
        try:
            fields = self._get_jira_json("/rest/api/3/field")
            self.jira_field_names = {
                field.get("id"): field.get("name")
                for field in fields
                if field.get("id") and field.get("name")
            }
        except Exception as e:
            print(f"Failed to fetch Jira field names: {e}")
            self.jira_field_names = {}
        return self.jira_field_names

    def fetch_space_name(self, space_key: str) -> str:
        try:
            data = self._get_json(f"{self.url}/rest/api/space/{space_key}")
            return data.get("name") or space_key
        except Exception as e:
            print(f"Failed to fetch Confluence space name for {space_key}: {e}")
            return space_key

    def search_spaces(self, query: str, limit: int = 10):
        normalized_query = query.strip().lower()
        if not normalized_query:
            return []

        url = f"{self.url}/api/v2/spaces"
        params = {"limit": 100, "status": "current"}
        matches = []

        while url and len(matches) < limit:
            data = self._get_json(url, params=params)
            for space in data.get("results", []):
                key = space.get("key") or ""
                name = space.get("name") or key
                if normalized_query not in key.lower() and normalized_query not in name.lower():
                    continue
                matches.append({"key": key, "name": name, "source_type": "confluence"})
                if len(matches) >= limit:
                    break

            next_path = (data.get("_links") or {}).get("next")
            url = self._absolute_url(next_path) if next_path else None
            params = None

        return matches

    def _absolute_url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http"):
            return path_or_url
        if path_or_url.startswith("/wiki"):
            return f"{self.site_url}{path_or_url}"
        return f"{self.url}/{path_or_url.lstrip('/')}"

    def image_filename_from_url(self, image_url: str) -> str:
        path = urlparse(image_url).path
        filename = os.path.basename(path)
        return unquote(filename) if filename else ""

    def nearby_text(self, element) -> str:
        container = element.find_parent(["p", "li", "td", "th", "figure", "div"])
        if not container:
            container = element.parent
        if not container:
            return ""

        text = container.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:1000]

    def clean_inline_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip()

    def extract_jira_issue_key(self, text: str) -> str | None:
        match = self.JIRA_ISSUE_KEY_PATTERN.search(text or "")
        return match.group(0) if match else None

    def resolve_jira_issue_label(self, issue_key: str) -> str:
        if issue_key in self.jira_issue_cache:
            return self.jira_issue_cache[issue_key]

        label = issue_key
        try:
            data = self._get_jira_json(f"/rest/api/3/issue/{issue_key}", params={"fields": "*all"})
            fields = data.get("fields") or {}
            summary = self.clean_inline_text(fields.get("summary") or "")
            status = self.clean_inline_text((fields.get("status") or {}).get("name") or "")
            custom_fields = format_jira_custom_fields(fields, self.fetch_jira_field_names())
            if summary and status:
                label = f"{issue_key}: {summary} ({status})"
            elif summary:
                label = f"{issue_key}: {summary}"
            if custom_fields:
                label = f"{label}; " + "; ".join(custom_fields)
        except Exception as e:
            print(f"Failed to resolve Jira issue {issue_key}: {e}")

        self.jira_issue_cache[issue_key] = label
        return label

    def resolve_jira_project_label(self, project_key: str) -> str:
        if project_key in self.jira_project_cache:
            return self.jira_project_cache[project_key]

        label = project_key
        try:
            data = self._get_jira_json(f"/rest/api/3/project/{project_key}")
            name = self.clean_inline_text(data.get("name") or "")
            if name:
                label = f"{name} ({project_key})"
        except Exception as e:
            print(f"Failed to resolve Jira project {project_key}: {e}")

        self.jira_project_cache[project_key] = label
        return label

    def extract_jira_link_label(self, element) -> str:
        href = element.get("href") or ""
        text = self.clean_inline_text(element.get_text(" ", strip=True))
        title = self.clean_inline_text(element.get("title") or element.get("aria-label") or "")
        data_issue_key = self.clean_inline_text(element.get("data-issue-key") or "")
        candidates = " ".join([data_issue_key, text, title, href])
        issue_key = data_issue_key or self.extract_jira_issue_key(candidates)

        if issue_key:
            if text and text != issue_key and not text.startswith("http"):
                return f"[Related Jira issue: {text}]"
            return f"[Related Jira issue: {self.resolve_jira_issue_label(issue_key)}]"

        project_match = re.search(r"/projects/([A-Z][A-Z0-9]+)(?:/|$)", href)
        if project_match:
            project_key = project_match.group(1)
            if text and not text.startswith("http"):
                return f"[Related Jira project: {text}]"
            return f"[Related Jira project: {self.resolve_jira_project_label(project_key)}]"

        if text and not text.startswith("http"):
            return f"[Related Jira link: {text}]"
        return ""

    def preprocess_html_rich_content(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")

        for jira in soup.find_all(["a", "span"], class_=["confluence-jim-macro", "jira-issue"]):
            label = self.extract_jira_link_label(jira)
            if label:
                jira.replace_with(f" {label} ")

        for jira_link in soup.find_all("a", href=re.compile(r"/jira/|atlassian\.net/jira/")):
            label = self.extract_jira_link_label(jira_link)
            if label:
                jira_link.replace_with(f" {label} ")

        for table in soup.find_all("table"):
            if self.is_jira_issue_table(table):
                table.decompose()
                continue

            table_text = self.table_to_text(table)
            if table_text:
                pre = soup.new_tag("pre")
                pre["data-metsabrain-table"] = "true"
                pre.string = table_text
                table.replace_with(pre)

        return str(soup)

    def normalize_cell_text(self, cell) -> str:
        text = cell.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text.replace("|", "\\|")

    def is_jira_issue_table(self, table) -> bool:
        rows = []
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if cells:
                rows.append([self.normalize_cell_text(cell) for cell in cells])

        if len(rows) < 2:
            return False

        headers = [header.strip().lower() for header in rows[0]]
        header_set = set(headers)
        has_jira_columns = {"key", "summary", "status"}.issubset(header_set)
        issue_key_count = 0
        issue_key_pattern = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")

        for row in rows[1:]:
            issue_key_count += sum(1 for cell in row if issue_key_pattern.search(cell))

        return has_jira_columns and issue_key_count > 0

    def table_to_text(self, table) -> str:
        rows = []
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if cells:
                rows.append([self.normalize_cell_text(cell) for cell in cells])

        if not rows:
            return ""

        column_count = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (column_count - len(row)) for row in rows]

        first_row = table.find("tr")
        has_header = bool(first_row and first_row.find_all("th"))
        if has_header:
            headers = normalized_rows[0]
            data_rows = normalized_rows[1:]
        else:
            headers = [f"Column {index + 1}" for index in range(column_count)]
            data_rows = normalized_rows

        markdown_rows = [
            "[Table start]",
            f"Rows: {len(data_rows)}",
            f"Columns: {', '.join(headers)}",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * column_count) + " |",
        ]

        for row in data_rows:
            markdown_rows.append("| " + " | ".join(row) + " |")

        markdown_rows.append("[Table end]")
        return "\n".join(markdown_rows)

    def split_text_preserving_tables(self, text: str):
        table_pattern = re.compile(r"\[Table start\].*?\[Table end\]", re.DOTALL)
        chunks = []
        cursor = 0

        for match in table_pattern.finditer(text):
            before = text[cursor : match.start()].strip()
            if before:
                chunks.extend(self.text_splitter.split_text(before))

            table_block = match.group(0).strip()
            if table_block:
                chunks.extend(self.split_table_block(table_block))

            cursor = match.end()

        after = text[cursor:].strip()
        if after:
            chunks.extend(self.text_splitter.split_text(after))

        return chunks

    def split_table_block(self, table_block: str):
        if len(table_block) <= self.max_chunk_chars:
            return [table_block]

        lines = table_block.splitlines()
        separator_index = next((i for i, line in enumerate(lines) if line.strip().startswith("| ---")), None)
        if separator_index is None:
            return self.text_splitter.split_text(table_block)

        prefix = lines[: separator_index + 1]
        row_lines = [line for line in lines[separator_index + 1 :] if line.strip().startswith("|")]
        chunks = []
        current_rows = []

        def flush_rows():
            if not current_rows:
                return
            table_lines = prefix + current_rows + ["[Table end]"]
            chunks.append("\n".join(table_lines))
            current_rows.clear()

        for row in row_lines:
            candidate = "\n".join(prefix + current_rows + [row, "[Table end]"])
            if len(candidate) > self.max_chunk_chars and current_rows:
                flush_rows()

            single_row_candidate = "\n".join(prefix + [row, "[Table end]"])
            if len(single_row_candidate) > self.max_chunk_chars:
                chunks.extend(self.text_splitter.split_text(single_row_candidate))
            else:
                current_rows.append(row)

        flush_rows()
        return chunks

    def extract_table_blocks(self, html_content: str):
        soup = BeautifulSoup(html_content, "html.parser")
        blocks = []
        for pre in soup.find_all("pre", attrs={"data-metsabrain-table": "true"}):
            text = pre.get_text("\n", strip=True)
            if "[Table start]" in text and "[Table end]" in text:
                blocks.append(text)
        return blocks

    def ensure_table_blocks_in_markdown(self, markdown_content: str, processed_html: str) -> str:
        table_blocks = self.extract_table_blocks(processed_html)
        if not table_blocks:
            return markdown_content

        missing_blocks = [block for block in table_blocks if block not in markdown_content]
        if not missing_blocks:
            return markdown_content

        return markdown_content.rstrip() + "\n\n" + "\n\n".join(missing_blocks)

    def fetch_page_adf_fallback_markdown(self, item) -> str:
        if not item.get("id"):
            return ""

        try:
            page = self._get_json(
                f"{self.url}/api/v2/pages/{item['id']}",
                params={"body-format": "atlas_doc_format"},
            )
            adf_value = ((page.get("body") or {}).get("atlas_doc_format") or {}).get("value")
            if not adf_value:
                return ""
            content = self.adf_to_markdown(json.loads(adf_value)).strip()
            return f"# {item.get('title')}\n\n{content}" if content else ""
        except Exception as e:
            print(f"Page ADF fallback fetch failed for {item.get('id')}: {e}")
            return ""

    def adf_to_markdown(self, value: Any) -> str:
        if not value:
            return ""
        if isinstance(value, list):
            return "".join(self.adf_to_markdown(item) for item in value)
        if not isinstance(value, dict):
            return str(value)

        node_type = value.get("type")
        if node_type == "text":
            return value.get("text", "")
        if node_type == "hardBreak":
            return "\n"
        if node_type == "table":
            return self.adf_table_to_text(value) + "\n\n"

        content = self.adf_to_markdown(value.get("content", []))
        if node_type == "paragraph":
            return f"{content.strip()}\n\n" if content.strip() else ""
        if node_type == "heading":
            level = value.get("attrs", {}).get("level", 1)
            return f"{'#' * level} {content.strip()}\n\n" if content.strip() else ""
        if node_type == "listItem":
            return f"- {content.strip()}\n" if content.strip() else ""
        if node_type in {"bulletList", "orderedList"}:
            return f"{content}\n"
        if node_type == "codeBlock":
            return f"```\n{content.strip()}\n```\n\n" if content.strip() else ""
        return content

    def adf_table_to_text(self, table: dict[str, Any]) -> str:
        rows = []
        has_header = False
        for row in table.get("content", []):
            if row.get("type") != "tableRow":
                continue
            cells = []
            for cell in row.get("content", []):
                if cell.get("type") not in {"tableCell", "tableHeader"}:
                    continue
                has_header = has_header or (not rows and cell.get("type") == "tableHeader")
                text = self.adf_to_markdown(cell.get("content", [])).strip().replace("\n", " ").replace("|", "\\|")
                cells.append(text)
            if cells:
                rows.append(cells)

        if not rows:
            return ""

        column_count = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (column_count - len(row)) for row in rows]
        if has_header:
            headers = normalized_rows[0]
            data_rows = normalized_rows[1:]
        else:
            headers = [f"Column {index + 1}" for index in range(column_count)]
            data_rows = normalized_rows

        markdown_rows = [
            "[Table start]",
            f"Rows: {len(data_rows)}",
            f"Columns: {', '.join(headers)}",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * column_count) + " |",
        ]
        markdown_rows.extend("| " + " | ".join(row) + " |" for row in data_rows)
        markdown_rows.append("[Table end]")
        return "\n".join(markdown_rows)

    def extract_image_documents(self, html_content: str, item, space_key: str, space_name: str, breadcrumb: str):
        soup = BeautifulSoup(html_content, "html.parser")
        documents = []
        seen = set()
        page_id = item.get("id")
        title = item.get("title")
        page_url = self.content_url(item)

        for index, image in enumerate(soup.find_all(["img", "ac:image"])):
            attachment = image.find("ri:attachment") if image.name == "ac:image" else None
            image_url = image.get("src") or image.get("data-image-src") or image.get("data-linked-resource-download-url") or ""
            if image_url:
                image_url = self._absolute_url(image_url)

            filename = (
                image.get("data-linked-resource-default-alias")
                or image.get("data-filename")
                or image.get("title")
                or image.get("alt")
                or (attachment.get("ri:filename") if attachment else "")
                or self.image_filename_from_url(image_url)
            )
            alt_text = image.get("alt") or image.get("title") or filename
            context_text = self.nearby_text(image)
            dedupe_key = image_url or f"{page_id}:{filename}:{index}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            if not image_url and not filename and not alt_text:
                continue

            lines = [
                "Content type: image",
                f"Parent page: {title}",
                f"Space: {space_name} ({space_key})",
                f"Document location: {breadcrumb}",
            ]
            if filename:
                lines.append(f"Filename: {filename}")
            if alt_text:
                lines.append(f"Alt text: {alt_text}")
            if image_url:
                lines.append(f"Image URL: {image_url}")
            if context_text:
                lines.extend(["", "Surrounding text:", context_text])

            documents.append(
                Document(
                    page_content="\n".join(lines),
                    metadata={
                        "page_id": page_id,
                        "title": filename or f"{title} image {index + 1}",
                        "url": page_url,
                        "image_url": image_url,
                        "space": space_key,
                        "space_name": space_name,
                        "breadcrumb": breadcrumb,
                        "content_type": "image",
                        "source_type": "confluence",
                        "parent_title": title,
                        "chunk_index": index,
                    },
                )
            )

        return documents

    def fetch_pages(self, space_key: str):
        cql_pages = self._fetch_cql_content(
            cql=f"space = '{space_key}' AND type = page",
            expand="body.export_view,body.storage,version,ancestors",
            label="pages",
        )
        descendant_pages = self.fetch_descendant_pages(cql_pages)
        merged_pages = self.merge_content_by_id(cql_pages, descendant_pages)

        cql_ids = {page.get("id") for page in cql_pages if page.get("id")}
        descendant_ids = {page.get("id") for page in descendant_pages if page.get("id")}
        missing_from_cql = descendant_ids - cql_ids
        print(
            "Confluence page completeness: "
            f"cql={len(cql_pages)} descendants={len(descendant_pages)} "
            f"merged={len(merged_pages)} descendant_only={len(missing_from_cql)}"
        )
        if missing_from_cql:
            print(f"Pages discovered only via descendants: {sorted(missing_from_cql)}")

        return merged_pages

    def merge_content_by_id(self, primary_items, fallback_items):
        merged = {}
        for item in fallback_items:
            item_id = item.get("id")
            if item_id:
                merged[item_id] = item
        for item in primary_items:
            item_id = item.get("id")
            if item_id:
                merged[item_id] = item
        return list(merged.values())

    def fetch_descendant_pages(self, pages):
        roots = [page for page in pages if page.get("id") and not page.get("ancestors")]
        if not roots and pages:
            roots = [page for page in pages if page.get("id")]

        all_descendants = []
        seen_ids = set()
        print(f"Searching Confluence descendants from {len(roots)} root pages.")

        for root in roots:
            for page in self.fetch_page_descendants(root["id"]):
                page_id = page.get("id")
                if page_id and page_id not in seen_ids:
                    seen_ids.add(page_id)
                    all_descendants.append(page)

        return all_descendants

    def fetch_page_descendants(self, page_id: str):
        return self._fetch_descendant_content(
            page_id=page_id,
            content_type="page",
            expand="body.export_view,body.storage,version,ancestors",
            label=f"page descendants for {page_id}",
        )

    def fetch_databases(self, space_key: str):
        cql = f"space = '{space_key}' AND type = database"
        databases = self._fetch_cql_content(
            cql=cql,
            expand=None,
            label="databases",
        )
        if not databases:
            databases = self._fetch_cql_search(cql, label="databases")

        enriched = []
        for database in databases:
            database_id = database.get("id")
            if not database_id:
                continue
            details = self.fetch_database_detail(database_id)
            if details:
                merged = {**database, **details}
                merged.setdefault("id", database_id)
                merged.setdefault("type", "database")
                enriched.append(merged)
            else:
                enriched.append(database)
        return enriched

    def fetch_database_detail(self, database_id: str):
        url = f"{self.url}/api/v2/databases/{database_id}"
        params = {
            "include-direct-children": "true",
            "include-properties": "true",
            "include-operations": "true",
        }
        try:
            return self._get_json(url, params=params)
        except Exception as e:
            print(f"Database detail fetch failed for {database_id}: {e}")
            return None

    def _fetch_cql_content(self, cql: str, expand: str | None, label: str):
        url = f"{self.url}/rest/api/content/search"
        params = {"cql": cql, "limit": 50}
        if expand:
            params["expand"] = expand
        all_results = []

        print(f"Searching Confluence {label}: {cql}")

        while url:
            try:
                data = self._get_json(url, params=params)
            except Exception as e:
                print(f"Fetch {label} failed: {e}")
                break

            all_results.extend(data.get("results", []))
            next_path = data.get("_links", {}).get("next")
            url = self._absolute_url(next_path) if next_path else None
            params = None

        return all_results

    def _fetch_cql_search(self, cql: str, label: str):
        url = f"{self.url}/rest/api/search"
        params = {"cql": cql, "limit": 50}
        all_results = []

        print(f"Searching Confluence {label} via fallback search API: {cql}")

        while url:
            try:
                data = self._get_json(url, params=params)
            except Exception as e:
                print(f"Fallback fetch {label} failed: {e}")
                break

            for result in data.get("results", []):
                content = result.get("content") or result
                if content.get("type") == "database":
                    all_results.append(content)

            next_path = data.get("_links", {}).get("next")
            url = self._absolute_url(next_path) if next_path else None
            params = None

        return all_results

    def _fetch_descendant_content(self, page_id: str, content_type: str, expand: str | None, label: str):
        url = f"{self.url}/rest/api/content/{page_id}/descendant/{content_type}"
        params = {"limit": 25}
        if expand:
            params["expand"] = expand
        all_results = []

        while url:
            try:
                data = self._get_json(url, params=params)
            except Exception as e:
                print(f"Fetch {label} failed: {e}")
                break

            all_results.extend(data.get("results", []))
            next_path = data.get("_links", {}).get("next")
            url = self._absolute_url(next_path) if next_path else None
            params = None

        return all_results

    def fetch_all_content(self, space_key: str):
        print(f"{ingest_label('confluence', space_key, self.last_source_name)}[fetch-start]")
        pages = self.fetch_pages(space_key)
        databases = self.fetch_databases(space_key)
        print(f"{ingest_label('confluence', space_key, self.last_source_name)}[fetch-complete] pages={len(pages)} databases={len(databases)}")
        return pages + databases

    def build_page_documents(self, item, space_key: str, space_name: str):
        title = item.get("title")
        item_id = item.get("id")
        content_type = item.get("type", "page")
        body_data = item.get("body", {})
        content_html = body_data.get("export_view", {}).get("value") or body_data.get("storage", {}).get("value", "")

        ancestors = item.get("ancestors", [])
        breadcrumb = " > ".join([a.get("title") for a in ancestors] + [title])
        prefix = f"Content type: {content_type}\nSpace: {space_name} ({space_key})\nDocument location: {breadcrumb}\n\n"
        markdown_content = self.fetch_page_adf_fallback_markdown(item) if not content_html else None
        if not content_html and not markdown_content:
            return []

        processed_html = self.preprocess_html_rich_content(content_html) if content_html else ""
        if markdown_content is None and not self._converter_unavailable:
            with tempfile.NamedTemporaryFile(suffix=".html", mode="w", encoding="utf-8", delete=False) as f:
                f.write(f"<html><body><h1>{title}</h1>{processed_html}</body></html>")
                temp_path = f.name

            try:
                result = self.converter.convert(temp_path)
                markdown_content = result.document.export_to_markdown()
            except Exception as e:
                print(f"Docling conversion failed. Falling back to BeautifulSoup text extraction: {e}")
                self._converter_unavailable = True
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        if markdown_content is None:
            markdown_content = self.html_to_text(title, processed_html)

        if processed_html:
            markdown_content = self.ensure_table_blocks_in_markdown(markdown_content, processed_html)
        raw_chunks = self.split_text_preserving_tables(markdown_content)
        documents = []
        #documents.extend(self.extract_image_documents(content_html, item, space_key, space_name, breadcrumb))
        for i, chunk in enumerate(raw_chunks):
            enriched_content = f"{prefix}Chunk: {i + 1}/{len(raw_chunks)}\n\nContent:\n{chunk}"
            documents.append(
                Document(
                    page_content=enriched_content,
                    metadata={
                        "page_id": item_id,
                        "title": title,
                        "url": self.content_url(item),
                        "space": space_key,
                        "space_name": space_name,
                        "breadcrumb": breadcrumb,
                        "content_type": content_type,
                        "source_type": "confluence",
                        "chunk_index": i,
                    },
                )
            )
        return documents

    def html_to_text(self, title: str, html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        lines = [f"# {title}", ""]
        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"]):
            separator = "\n" if element.name == "pre" else " "
            text = element.get_text(separator, strip=True)
            if not text:
                continue
            if element.name in {"h1", "h2", "h3", "h4"}:
                level = int(element.name[1])
                lines.append(f"{'#' * level} {text}")
            elif element.name == "li":
                lines.append(f"- {text}")
            else:
                lines.append(text)

        if len(lines) <= 2:
            text = soup.get_text("\n", strip=True)
            if text:
                lines.append(text)

        return "\n\n".join(lines)

    def build_database_documents(self, item, space_key: str, space_name: str):
        title = item.get("title") or "Untitled database"
        item_id = item.get("id")
        ancestors = item.get("ancestors", [])
        breadcrumb = " > ".join([a.get("title") for a in ancestors if a.get("title")] + [title])
        direct_children = item.get("directChildren") or item.get("children") or []
        version = item.get("version") or {}
        parent_id = item.get("parentId") or item.get("parent", {}).get("id")
        parent_type = item.get("parentType") or item.get("parent", {}).get("type")

        lines = [
            "Content type: database",
            f"Title: {title}",
            f"Database ID: {item_id}",
            f"Space: {space_name} ({space_key})",
            f"Document location: {breadcrumb}",
        ]
        if parent_id:
            lines.append(f"Parent: {parent_type or 'content'} {parent_id}")
        if version:
            lines.append(f"Version: {version.get('number', 'unknown')}")
            if version.get("createdAt"):
                lines.append(f"Version created at: {version['createdAt']}")
        if direct_children:
            lines.append("Direct children:")
            for child in direct_children:
                if isinstance(child, dict):
                    child_title = child.get("title") or child.get("id")
                    child_type = child.get("type", "content")
                else:
                    child_title = str(child)
                    child_type = "content"
                lines.append(f"- {child_type}: {child_title}")

        lines.append(
            "Note: Confluence's public REST API exposes database metadata, but not database rows/entries for indexing."
        )

        content = "\n".join(lines)
        chunks = self.text_splitter.split_text(content)

        return [
            Document(
                page_content=f"Chunk: {i + 1}/{len(chunks)}\n\nContent:\n{chunk}",
                metadata={
                    "page_id": item_id,
                    "title": title,
                    "url": self.content_url(item),
                    "space": space_key,
                    "space_name": space_name,
                    "breadcrumb": breadcrumb,
                    "content_type": "database",
                    "source_type": "confluence",
                    "chunk_index": i,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

    def content_url(self, item):
        links = item.get("_links") or {}
        webui = links.get("webui")
        base = links.get("base") or self.site_url
        if webui:
            return self._absolute_url(webui)
        if item.get("type") == "database":
            return f"{base}/wiki/spaces/{item.get('spaceKey', '')}/database/{item.get('id')}"
        return f"{self.url}/pages/viewpage.action?pageId={item.get('id')}"

    def process_and_save(self, space_key: str, replace_existing: bool = True):
        space_name = self.fetch_space_name(space_key)
        self.last_source_name = space_name
        print(f"{ingest_label('confluence', space_key, space_name)}[start] replace_existing={replace_existing}")
        contents = self.fetch_all_content(space_key)
        if not contents:
            print(f"{ingest_label('confluence', space_key, space_name)}[complete] chunks=0 reason=no-content")
            return 0

        documents = []
        skipped_pages = []
        for item in contents:
            if item.get("type") == "database":
                documents.extend(self.build_database_documents(item, space_key, space_name))
            else:
                page_documents = self.build_page_documents(item, space_key, space_name)
                if page_documents:
                    documents.extend(page_documents)
                else:
                    skipped_pages.append({"id": item.get("id"), "title": item.get("title"), "type": item.get("type")})

        if skipped_pages:
            print(f"{ingest_label('confluence', space_key, space_name)}[skip] items={len(skipped_pages)} sample={skipped_pages[:20]}")

        if documents:
            print(f"{ingest_label('confluence', space_key, space_name)}[index-start] chunks={len(documents)}")
            save_source_documents(
                self.qdrant_host,
                "confluence",
                space_key,
                documents,
                self.embeddings,
                replace_existing,
            )
            print(f"{ingest_label('confluence', space_key, space_name)}[index-complete] chunks={len(documents)}")
        return len(documents)


class JiraIngestor:
    def __init__(self):
        self.url = normalize_atlassian_site_url(os.getenv("JIRA_URL") or os.getenv("CONFLUENCE_URL", ""))
        self.email = os.getenv("JIRA_EMAIL", os.getenv("CONFLUENCE_EMAIL"))
        self.api_token = os.getenv("JIRA_API_TOKEN", os.getenv("CONFLUENCE_API_TOKEN"))
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
        self.max_issues = int(os.getenv("JIRA_INGEST_MAX_ISSUES", "1000"))
        self.max_comments_per_issue = int(os.getenv("JIRA_INGEST_MAX_COMMENTS_PER_ISSUE", "20"))
        self.request_timeout = float(os.getenv("JIRA_REQUEST_TIMEOUT_SECONDS", "90"))
        self.last_source_name = None
        self.field_names = None

        self.embeddings = OllamaEmbeddings(base_url=self.ollama_host, model=self.embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def get_auth_header(self):
        headers = {"Accept": "application/json"}
        if self.email and self.api_token:
            credentials = f"{self.email}:{self.api_token}"
            encoded_auth = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded_auth}"
        return headers

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.get(url, headers=self.get_auth_header(), params=params, timeout=self.request_timeout)
        if response.status_code >= 400:
            detail = response.text[:500]
            raise RuntimeError(f"Jira API error {response.status_code} for {response.url}: {detail}")
        return response.json()

    def fetch_field_names(self) -> dict[str, str]:
        if self.field_names is not None:
            return self.field_names
        try:
            fields = self._get_json(f"{self.url}/rest/api/3/field")
            self.field_names = {
                field.get("id"): field.get("name")
                for field in fields
                if field.get("id") and field.get("name")
            }
        except Exception as e:
            print(f"{ingest_label('jira', 'fields', self.last_source_name)}[field-fetch-failed] error={e}")
            self.field_names = {}
        return self.field_names

    def search_projects(self, query: str, limit: int = 10):
        data = self._get_json(
            f"{self.url}/rest/api/3/project/search",
            params={"query": query.strip(), "maxResults": limit, "orderBy": "name"},
        )
        return [
            {
                "key": project.get("key"),
                "name": project.get("name") or project.get("key"),
                "source_type": "jira",
            }
            for project in data.get("values", [])
            if project.get("key")
        ]

    def fetch_issues(self, project_key: str):
        url = f"{self.url}/rest/api/3/search/jql"
        params = {
            "jql": f"project = {project_key} ORDER BY updated DESC",
            "maxResults": 100,
            "fields": "*all",
        }
        issues = []

        print(f"{ingest_label('jira', project_key, self.last_source_name)}[fetch-start]")

        while True:
            data = self._get_json(url, params=params)
            issues.extend(data.get("issues", []))
            if len(issues) >= self.max_issues:
                return issues[: self.max_issues]

            next_page_token = data.get("nextPageToken")
            if not next_page_token or data.get("isLast"):
                return issues

            params["nextPageToken"] = next_page_token

    def adf_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(filter(None, [self.adf_to_text(item) for item in value]))
        if not isinstance(value, dict):
            return str(value)

        node_type = value.get("type")
        if node_type == "text":
            return value.get("text", "")

        parts = [self.adf_to_text(item) for item in value.get("content", [])]
        text = " ".join(part for part in parts if part).strip()
        if node_type in {"paragraph", "heading", "listItem"}:
            return text
        if node_type in {"bulletList", "orderedList"}:
            return "\n".join(f"- {part}" for part in parts if part)
        return text

    def format_comments(self, comment_data: dict[str, Any]):
        comments = comment_data.get("comments", []) if isinstance(comment_data, dict) else []
        lines = []

        for comment in comments[: self.max_comments_per_issue]:
            body = self.adf_to_text(comment.get("body"))
            if not body:
                continue
            author = comment.get("author") or {}
            author_name = author.get("displayName") or "unknown"
            created = comment.get("created") or "unknown"
            lines.append(f"- {author_name} at {created}: {body}")

        return lines

    def issue_overview(self, issue: dict[str, Any], project_key: str):
        fields = issue.get("fields") or {}
        key = issue.get("key")
        summary = fields.get("summary") or key or "Untitled issue"
        project = fields.get("project") or {}
        project_name = project.get("name") or project_key
        status = fields.get("status") or {}
        assignee = fields.get("assignee") or {}
        reporter = fields.get("reporter") or {}
        custom_fields = jira_custom_field_map(fields, self.fetch_field_names())
        date_fields = jira_date_fields(custom_fields)
        due_date = fields.get("duedate") or next(iter(date_fields.values()), None)
        return {
            "key": key,
            "summary": summary,
            "project_name": project_name,
            "status_name": status.get("name", "unknown"),
            "status_category": normalize_jira_status_category(status),
            "assignee_name": assignee.get("displayName", "unassigned"),
            "reporter_name": reporter.get("displayName", "unknown"),
            "created": fields.get("created"),
            "updated": fields.get("updated"),
            "due_date": due_date,
            "custom_fields": custom_fields,
            "date_fields": date_fields,
        }

    def issue_to_documents(self, issue: dict[str, Any], project_key: str):
        fields = issue.get("fields") or {}
        overview = self.issue_overview(issue, project_key)
        key = overview["key"]
        summary = overview["summary"]
        project_name = overview["project_name"]
        issue_type = fields.get("issuetype") or {}
        status = fields.get("status") or {}
        priority = fields.get("priority") or {}
        components = fields.get("components") or []
        fix_versions = fields.get("fixVersions") or []
        labels = fields.get("labels") or []
        description = self.adf_to_text(fields.get("description"))
        comment_lines = self.format_comments(fields.get("comment") or {})
        custom_field_lines = [f"{name}: {value}" for name, value in overview["custom_fields"].items()]
        url = f"{self.url}/browse/{key}"
        breadcrumb = f"{project_name} > {key}"

        lines = [
            "Content type: jira_issue",
            f"Issue key: {key}",
            f"Title: {summary}",
            f"Project: {project_name}",
            f"Issue type: {issue_type.get('name', 'unknown')}",
            f"Status: {status.get('name', 'unknown')}",
            f"Status category: {overview['status_category']}",
            f"Priority: {priority.get('name', 'unknown')}",
            f"Assignee: {overview['assignee_name']}",
            f"Reporter: {overview['reporter_name']}",
            f"Created: {overview['created']}",
            f"Updated: {overview['updated']}",
        ]
        if overview["due_date"]:
            lines.append(f"Due date: {overview['due_date']}")
        if labels:
            lines.append(f"Labels: {', '.join(labels)}")
        if components:
            lines.append("Components: " + ", ".join(component.get("name", "") for component in components))
        if fix_versions:
            lines.append("Fix versions: " + ", ".join(version.get("name", "") for version in fix_versions))
        if custom_field_lines:
            lines.extend(["", f"Custom fields indexed: {len(custom_field_lines)}", *custom_field_lines])
        if description:
            lines.extend(["", "Description:", description])
        if comment_lines:
            lines.extend(["", f"Comments indexed: {len(comment_lines)}", *comment_lines])

        chunks = self.text_splitter.split_text("\n".join(lines))
        return [
            Document(
                page_content=f"Chunk: {i + 1}/{len(chunks)}\n\nContent:\n{chunk}",
                metadata={
                    "page_id": key,
                    "title": f"{key} {summary}",
                    "url": url,
                    "space": project_key,
                    "space_name": project_name,
                    "breadcrumb": breadcrumb,
                    "content_type": "jira_issue",
                    "source_type": "jira",
                    "issue_key": key,
                    "issue_status": overview["status_name"],
                    "issue_status_category": overview["status_category"],
                    "assignee": overview["assignee_name"],
                    "reporter": overview["reporter_name"],
                    "due_date": overview["due_date"],
                    "updated": overview["updated"],
                    "created": overview["created"],
                    "jira_custom_fields": overview["custom_fields"],
                    "jira_date_fields": overview["date_fields"],
                    "chunk_index": i,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

    def build_project_status_summary_documents(self, issues: list[dict[str, Any]], project_key: str, project_name: str):
        status_labels = {
            "todo": "To do",
            "in_progress": "In progress",
            "done": "Done",
        }
        grouped = {"todo": [], "in_progress": [], "done": []}
        for issue in issues:
            overview = self.issue_overview(issue, project_key)
            grouped.setdefault(overview["status_category"], []).append(overview)

        lines = [
            "Content type: jira_project_status_summary",
            f"Project: {project_name} ({project_key})",
            f"Total issues indexed: {len(issues)}",
            "Status category counts: "
            + ", ".join(f"{status_labels.get(category, category)}={len(items)}" for category, items in grouped.items()),
            "",
        ]

        for category in ("todo", "in_progress", "done"):
            items = sorted(grouped.get(category, []), key=lambda item: item.get("updated") or "", reverse=True)
            lines.append(f"{status_labels[category]} issues ({len(items)}):")
            if not items:
                lines.append("- None")
                lines.append("")
                continue
            for item in items:
                details = [
                    f"status={item['status_name']}",
                    f"assignee={item['assignee_name']}",
                    f"updated={item['updated']}",
                ]
                if item.get("due_date"):
                    details.append(f"due_date={item['due_date']}")
                lines.append(f"- {item['key']}: {item['summary']} ({'; '.join(details)})")
            lines.append("")

        chunks = self.text_splitter.split_text("\n".join(lines))
        return [
            Document(
                page_content=f"Chunk: {i + 1}/{len(chunks)}\n\nContent:\n{chunk}",
                metadata={
                    "page_id": f"{project_key}:status-summary",
                    "title": f"{project_key} Jira status summary",
                    "url": f"{self.url}/jira/software/projects/{project_key}/issues",
                    "space": project_key,
                    "space_name": project_name,
                    "breadcrumb": f"{project_name} > Jira status summary",
                    "content_type": "jira_project_status_summary",
                    "source_type": "jira",
                    "issue_status_category": "summary",
                    "jira_status_counts": {category: len(items) for category, items in grouped.items()},
                    "chunk_index": i,
                },
            )
            for i, chunk in enumerate(chunks)
        ]

    def process_and_save(self, project_key: str, replace_existing: bool = True):
        print(f"{ingest_label('jira', project_key, self.last_source_name)}[start] replace_existing={replace_existing}")
        issues = self.fetch_issues(project_key)
        if not issues:
            print(f"{ingest_label('jira', project_key, self.last_source_name)}[complete] chunks=0 reason=no-issues")
            return 0

        first_project = (issues[0].get("fields") or {}).get("project") or {}
        self.last_source_name = first_project.get("name") or project_key
        print(f"{ingest_label('jira', project_key, self.last_source_name)}[fetch-complete] issues={len(issues)}")

        documents = []
        for issue in issues:
            documents.extend(self.issue_to_documents(issue, project_key))
        documents.extend(self.build_project_status_summary_documents(issues, project_key, self.last_source_name))

        if documents:
            print(f"{ingest_label('jira', project_key, self.last_source_name)}[index-start] chunks={len(documents)}")
            save_source_documents(
                self.qdrant_host,
                "jira",
                project_key,
                documents,
                self.embeddings,
                replace_existing,
            )
            print(f"{ingest_label('jira', project_key, self.last_source_name)}[index-complete] chunks={len(documents)}")
        return len(documents)
