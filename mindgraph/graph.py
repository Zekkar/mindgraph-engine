"""
Wiki Knowledge Graph Engine
解析 wiki/ markdown → NetworkX 圖 → 關鍵字搜尋 + 圖共現加分 + Section 導航

社群偵測 / god nodes / 最短路徑已於 v0.3.0 移除：實測使用量為 0，
圖本身只保留給混合檢索的共現加分與 get_concept 的鄰居清單。
"""
import re
import math
import logging
import frontmatter
import networkx as nx
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# 共用標籤推測邊：只在 2..MAX_TAG_FANOUT 頁共用時連線；共用頁數過多的標籤會把全圖連成樞紐，
# 讓圖共現加分失去鑑別力（不變式 I-10）
MAX_TAG_FANOUT = 16


def category_from_path(md_path: Path, wiki_root: Path) -> str:
    """由 wiki markdown 路徑推出知識種類：取所在資料夾名稱
    （wiki/decisions/x.md → decisions）；位於 wiki 根目錄回傳 'root'。
    圖引擎、向量庫、時間遞減、hook 都依這一個規則判定種類。"""
    try:
        rel = Path(md_path).relative_to(wiki_root)
    except ValueError:
        rel = Path(md_path)
    return rel.parent.name if rel.parent.parts else "root"


class WikiGraphEngine:
    def __init__(self, wiki_root: str):
        self.wiki_root = Path(wiki_root)
        self.graph = nx.Graph()
        self.pages = {}
        self.tfidf_index = defaultdict(dict)
        self._last_build = None
        self._file_mtimes = {}

    def build(self):
        self._scan_wiki_files()
        self._build_graph()
        self._build_search_index()
        self._last_build = datetime.now()
        stats = self.get_stats()
        logger.info(
            "graph built",
            extra={
                "total_pages": stats["total_pages"],
                "total_edges": stats["total_edges"],
            },
        )

    def needs_rebuild(self) -> bool:
        if not self._last_build:
            return True
        current = {str(p): p.stat().st_mtime for p in self.wiki_root.rglob("*.md")}
        if set(current) != set(self._file_mtimes):
            return True  # a file was added or deleted
        return any(self._file_mtimes.get(k) != mt for k, mt in current.items())

    def build_incremental(self):
        """只重新解析變動/新增的檔案（依 mtime），移除已刪除頁面，再從 self.pages 重建
        記憶體圖/索引。避免每次變動都重讀並解析全部 md——檔案多或大時這是 ensure_fresh 的主要成本。"""
        current = {str(p): p for p in self.wiki_root.rglob("*.md")}
        for known in [k for k in self._file_mtimes if k not in current]:
            self.pages.pop(Path(known).stem, None)
            self._file_mtimes.pop(known, None)
        changed = 0
        for path_str, md_path in current.items():
            mtime = md_path.stat().st_mtime
            if self._file_mtimes.get(path_str) == mtime:
                continue
            try:
                self.pages[md_path.stem] = self._load_page(md_path)
                self._file_mtimes[path_str] = mtime
                changed += 1
            except Exception as e:
                logger.warning("skip markdown file", extra={"path": str(md_path), "error": str(e)})
        self._build_graph()
        self._build_search_index()
        self._last_build = datetime.now()
        logger.info("graph incrementally rebuilt", extra={"changed_files": changed})

    def ensure_fresh(self):
        if not self._last_build:
            self.build()
        elif self.needs_rebuild():
            self.build_incremental()

    # === Build Phase ===

    def _load_page(self, md_path) -> dict:
        post = frontmatter.load(str(md_path))
        return {
            'content': post.content,
            'metadata': dict(post.metadata),
            'path': str(md_path.relative_to(self.wiki_root.parent)),
            'category': category_from_path(md_path, self.wiki_root),
        }

    def _scan_wiki_files(self):
        self.pages = {}
        self._file_mtimes = {}
        for md_path in self.wiki_root.rglob("*.md"):
            try:
                self.pages[md_path.stem] = self._load_page(md_path)
                self._file_mtimes[str(md_path)] = md_path.stat().st_mtime
            except Exception as e:
                logger.warning("skip markdown file", extra={"path": str(md_path), "error": str(e)})

    def _build_graph(self):
        self.graph = nx.Graph()

        for name, page in self.pages.items():
            self.graph.add_node(
                name,
                category=page['category'],
                path=page['path'],
                node_type=page['metadata'].get('type', 'unknown'),
                tags=page['metadata'].get('tags', []),
            )

        # EXTRACTED edges: frontmatter [[related]]
        for name, page in self.pages.items():
            for rel in page['metadata'].get('related', []):
                rel_name = re.sub(r'\[\[|\]\]', '', str(rel))
                if rel_name in self.pages:
                    self.graph.add_edge(name, rel_name,
                                        confidence='EXTRACTED', source='frontmatter')

            # EXTRACTED edges: content [[wikilinks]]
            for link in re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]', page['content']):
                if link in self.pages and link != name and not self.graph.has_edge(name, link):
                    self.graph.add_edge(name, link,
                                        confidence='EXTRACTED', source='wikilink')

        # INFERRED edges: shared non-generic tags
        generic_tags = {'LLM-synthesized', '概念', '策略', '系統', '架構', 'raw'}
        tag_to_pages = defaultdict(set)
        for name, page in self.pages.items():
            for tag in page['metadata'].get('tags', []):
                if tag not in generic_tags:
                    tag_to_pages[tag].add(name)

        for tag, members in tag_to_pages.items():
            if 2 <= len(members) <= MAX_TAG_FANOUT:
                members_list = sorted(members)
                for i, p1 in enumerate(members_list):
                    for p2 in members_list[i + 1:]:
                        if not self.graph.has_edge(p1, p2):
                            self.graph.add_edge(p1, p2,
                                                confidence='INFERRED', source=f'tag:{tag}')

    def _build_search_index(self):
        doc_terms = {}
        doc_freq = Counter()
        for name, page in self.pages.items():
            text = f"{name} {page['content']} {' '.join(str(t) for t in page['metadata'].get('tags', []))}"
            terms = self._tokenize(text)
            tf = Counter(terms)
            doc_terms[name] = tf
            for t in set(terms):
                doc_freq[t] += 1

        n = len(self.pages)
        self.tfidf_index = defaultdict(dict)
        for name, tf in doc_terms.items():
            mx = max(tf.values()) if tf else 1
            for term, freq in tf.items():
                self.tfidf_index[term][name] = (freq / mx) * math.log(n / (1 + doc_freq[term]))

    @staticmethod
    def _tokenize(text: str) -> list:
        tokens = re.findall(r'[\w一-鿿㐀-䶿]+', text.lower())
        expanded = []
        for tok in tokens:
            expanded.append(tok)
            cjk = re.findall(r'[一-鿿㐀-䶿]', tok)
            for i in range(len(cjk) - 1):
                expanded.append(cjk[i] + cjk[i + 1])
        return expanded

    # === Query Methods ===

    def search(self, query: str, limit: int = 10) -> list:
        self.ensure_fresh()
        scores = Counter()
        for term in self._tokenize(query):
            for doc, sc in self.tfidf_index.get(term, {}).items():
                scores[doc] += sc

        results = []
        for name, score in scores.most_common(limit):
            page = self.pages[name]
            lines = [l.strip() for l in page['content'].split('\n')
                     if l.strip() and not l.startswith('#') and not l.startswith('---')]
            results.append({
                'name': name, 'path': page['path'], 'category': page['category'],
                'score': round(score, 3),
                'snippet': (lines[0][:200] if lines else ''),
                'tags': page['metadata'].get('tags', []),
            })
        return results

    def get_concept(self, name: str) -> dict | None:
        self.ensure_fresh()
        match = self.pages.get(name)
        if not match:
            for pn in self.pages:
                if name.lower() in pn.lower() or pn.lower() in name.lower():
                    match = self.pages[pn]
                    name = pn
                    break
        if not match:
            return None
        return {
            'name': name, 'content': match['content'], 'metadata': match['metadata'],
            'path': match['path'], 'category': match['category'],
            'neighbors': list(self.graph.neighbors(name)) if name in self.graph else [],
        }

    def neighbors(self, name: str) -> set:
        """概念在圖上的 1-hop 鄰居名稱；供混合檢索的圖共現加分使用。"""
        return set(self.graph.neighbors(name)) if name in self.graph else set()

    def get_stats(self) -> dict:
        conf = Counter(d.get('confidence', '?') for _, _, d in self.graph.edges(data=True))
        return {
            'total_pages': len(self.pages),
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'edge_confidences': dict(conf),
            'density': round(nx.density(self.graph), 4) if self.graph.number_of_nodes() > 1 else 0,
            'categories': dict(Counter(p['category'] for p in self.pages.values())),
            'last_build': self._last_build.isoformat() if self._last_build else None,
        }

    def get_health_stats(self) -> dict:
        """統計＋結構健康度（斷鏈、缺 frontmatter、孤立頁），REST 與 MCP 的 stats 共用。"""
        s = self.get_stats()
        d = self.get_diagnostics()
        s.update({"broken_link_count": d["broken_count"],
                  "no_frontmatter_count": d["no_frontmatter_count"],
                  "orphan_count": d["orphan_count"]})
        return s

    def get_diagnostics(self) -> dict:
        """擴充健康診斷（供 `mindgraph check` 用）：孤立頁面（degree 0、無連結）、
        斷掉的 wikilink（指向不存在的頁）、缺 frontmatter 的頁。檢出 wiki 知識庫的結構問題。"""
        self.ensure_fresh()
        orphans = sorted(n for n in self.graph.nodes if self.graph.degree(n) == 0)
        broken = []
        for name, page in self.pages.items():
            for link in re.findall(r'\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]', page['content']):
                if link not in self.pages:
                    broken.append({"page": name, "link": link})
        no_meta = sorted(n for n, p in self.pages.items() if not p['metadata'])
        return {
            "orphan_pages": orphans,
            "orphan_count": len(orphans),
            "broken_wikilinks": broken[:100],
            "broken_count": len(broken),
            "pages_without_frontmatter": no_meta,
            "no_frontmatter_count": len(no_meta),
        }

    # === Section Navigation（Hierarchical Chunking）===

    @staticmethod
    def _parse_sections(content: str) -> list[dict]:
        sections = []
        for line in content.split('\n'):
            m = re.match(r'^(#{2,4})\s+(.+)', line)
            if m:
                heading = m.group(2).strip()
                sections.append({'id': heading, 'level': len(m.group(1))})
        return sections

    def get_sections(self, name: str) -> dict | None:
        self.ensure_fresh()
        concept = self.get_concept(name)
        if not concept:
            hits = [n for n in self.pages if name.lower() in n.lower()]
            return {'error': f'找不到: {name}', 'suggestions': hits[:5]}

        sections = self._parse_sections(concept['content'])
        return {
            'name': concept['name'],
            'path': concept['path'],
            'section_count': len(sections),
            'sections': [
                {**s, 'address': f"{concept['name']}#{s['id']}"}
                for s in sections
            ],
        }

    def get_section(self, name: str, section_id: str) -> dict:
        self.ensure_fresh()
        concept = self.get_concept(name)
        if not concept:
            return {'error': f'找不到概念: {name}'}

        lines = concept['content'].split('\n')
        target_idx = None
        target_level = None

        for i, line in enumerate(lines):
            m = re.match(r'^(#{2,4})\s+(.+)', line)
            if m and m.group(2).strip() == section_id:
                target_idx, target_level = i, len(m.group(1))
                break

        if target_idx is None:
            for i, line in enumerate(lines):
                m = re.match(r'^(#{2,4})\s+(.+)', line)
                if m and section_id.lower() in m.group(2).lower():
                    target_idx, target_level = i, len(m.group(1))
                    break

        if target_idx is None:
            available = [s['id'] for s in self._parse_sections(concept['content'])]
            return {
                'error': f'找不到章節: {section_id}',
                'concept': concept['name'],
                'available_sections': available,
            }

        section_lines = [lines[target_idx]]
        for line in lines[target_idx + 1:]:
            m = re.match(r'^(#{1,4})\s+', line)
            if m and len(m.group(1)) <= target_level:
                break
            section_lines.append(line)

        return {
            'name': concept['name'],
            'section_id': section_id,
            'address': f"{concept['name']}#{section_id}",
            'level': target_level,
            'content': '\n'.join(section_lines).strip(),
        }
