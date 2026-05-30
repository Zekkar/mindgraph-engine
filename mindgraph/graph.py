"""
Wiki Knowledge Graph Engine
解析 wiki/ markdown → NetworkX 圖 → 搜尋 + 社群偵測 + Section 導航
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


class WikiGraphEngine:
    def __init__(self, wiki_root: str):
        self.wiki_root = Path(wiki_root)
        self.graph = nx.Graph()
        self.pages = {}
        self.tfidf_index = defaultdict(dict)
        self.communities = {}
        self.community_names = {}
        self._last_build = None
        self._file_mtimes = {}

    def build(self):
        self._scan_wiki_files()
        self._build_graph()
        self._detect_communities()
        self._build_search_index()
        self._last_build = datetime.now()
        stats = self.get_stats()
        logger.info(
            "graph built",
            extra={
                "total_pages": stats["total_pages"],
                "total_edges": stats["total_edges"],
                "communities": stats["communities"],
            },
        )

    def needs_rebuild(self) -> bool:
        if not self._last_build:
            return True
        for path in self.wiki_root.rglob("*.md"):
            mtime = path.stat().st_mtime
            if str(path) not in self._file_mtimes or self._file_mtimes[str(path)] != mtime:
                return True
        return False

    def ensure_fresh(self):
        if self.needs_rebuild():
            self.build()

    # === Build Phase ===

    def _scan_wiki_files(self):
        self.pages = {}
        self._file_mtimes = {}
        for md_path in self.wiki_root.rglob("*.md"):
            name = md_path.stem
            try:
                post = frontmatter.load(str(md_path))
                self.pages[name] = {
                    'content': post.content,
                    'metadata': dict(post.metadata),
                    'path': str(md_path.relative_to(self.wiki_root.parent)),
                    'category': md_path.parent.name if md_path.parent != self.wiki_root else 'root',
                }
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
            if 2 <= len(members) <= 6:
                members_list = sorted(members)
                for i, p1 in enumerate(members_list):
                    for p2 in members_list[i + 1:]:
                        if not self.graph.has_edge(p1, p2):
                            self.graph.add_edge(p1, p2,
                                                confidence='INFERRED', source=f'tag:{tag}')

    def _detect_communities(self):
        if len(self.graph.nodes) < 3:
            self.communities = {n: 0 for n in self.graph.nodes}
            self.community_names = {0: '全體'}
            return
        try:
            comms = nx.community.louvain_communities(self.graph, seed=42)
            self.communities = {}
            self.community_names = {}
            for i, members in enumerate(comms):
                for node in members:
                    self.communities[node] = i
                cats = Counter(self.pages[n]['category'] for n in members if n in self.pages)
                top_cat = cats.most_common(1)[0][0] if cats else '?'
                self.community_names[i] = f"{top_cat}-{i}"
        except Exception as e:
            logger.warning("community detection failed", extra={"error": str(e)})
            self.communities = {n: 0 for n in self.graph.nodes}

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
            'community': self.communities.get(name),
            'neighbors': list(self.graph.neighbors(name)) if name in self.graph else [],
        }

    def get_related(self, concept: str, depth: int = 2) -> dict:
        self.ensure_fresh()
        if concept not in self.graph:
            hits = [n for n in self.graph.nodes if concept.lower() in n.lower()]
            if hits:
                concept = hits[0]
            else:
                return {'error': f'找不到 "{concept}"',
                        'available': sorted(self.graph.nodes)[:20]}

        visited = {concept: 0}
        queue = [(concept, 0)]
        edges = []
        while queue:
            node, d = queue.pop(0)
            if d >= depth:
                continue
            for nb in self.graph.neighbors(node):
                ed = self.graph.edges[node, nb]
                edges.append({'from': node, 'to': nb,
                              'confidence': ed.get('confidence', ''),
                              'source': ed.get('source', '')})
                if nb not in visited:
                    visited[nb] = d + 1
                    queue.append((nb, d + 1))

        nodes = [{'name': n, 'distance': d, 'category': self.pages.get(n, {}).get('category', ''),
                  'community': self.communities.get(n)}
                 for n, d in sorted(visited.items(), key=lambda x: x[1])]
        return {'root': concept, 'depth': depth, 'nodes': nodes, 'edges': edges,
                'node_count': len(nodes), 'edge_count': len(edges)}

    def get_communities(self) -> list:
        self.ensure_fresh()
        buckets = defaultdict(list)
        for node, cid in self.communities.items():
            buckets[cid].append(node)
        return [{'id': cid, 'name': self.community_names.get(cid, ''),
                 'members': members, 'size': len(members)}
                for cid, members in sorted(buckets.items())]

    def get_stats(self) -> dict:
        conf = Counter(d.get('confidence', '?') for _, _, d in self.graph.edges(data=True))
        return {
            'total_pages': len(self.pages),
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'communities': len(set(self.communities.values())) if self.communities else 0,
            'edge_confidences': dict(conf),
            'density': round(nx.density(self.graph), 4) if self.graph.number_of_nodes() > 1 else 0,
            'categories': dict(Counter(p['category'] for p in self.pages.values())),
            'last_build': self._last_build.isoformat() if self._last_build else None,
        }

    def get_god_nodes(self, limit: int = 5) -> list:
        self.ensure_fresh()
        return [{'name': n, 'degree': d, 'category': self.pages.get(n, {}).get('category', ''),
                 'community': self.communities.get(n),
                 'neighbors': list(self.graph.neighbors(n))}
                for n, d in sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)[:limit]]

    def shortest_path(self, source: str, target: str) -> dict:
        self.ensure_fresh()
        for orig, name_ref in [(source, 'source'), (target, 'target')]:
            if orig not in self.graph:
                hits = [n for n in self.graph.nodes if orig.lower() in n.lower()]
                if hits:
                    if name_ref == 'source':
                        source = hits[0]
                    else:
                        target = hits[0]
        if source not in self.graph or target not in self.graph:
            return {'error': '找不到起點或終點'}
        try:
            path = nx.shortest_path(self.graph, source, target)
            edges = [{'from': path[i], 'to': path[i + 1],
                      **self.graph.edges[path[i], path[i + 1]]}
                     for i in range(len(path) - 1)]
            return {'source': source, 'target': target, 'path': path,
                    'length': len(path) - 1, 'edges': edges}
        except nx.NetworkXNoPath:
            return {'error': f'{source} 和 {target} 之間無路徑'}

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
