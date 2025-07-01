import re
import os
import time
import logging
import asyncio
import aiohttp
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import fitz  # PyMuPDF
import PyPDF2
import requests
from bs4 import BeautifulSoup
import networkx as nx
import validators
import textstat
from tqdm import tqdm
import trafilatura
from newspaper import Article
from markdownify import markdownify as md

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedLink:
    """Represents a link extracted from PDF or web content"""
    url: str
    text: str = ""
    page_number: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    context: str = ""
    parent_url: Optional[str] = None
    depth: int = 0


@dataclass
class ScrapedContent:
    """Represents scraped web content"""
    url: str
    title: str = ""
    content: str = ""
    images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    links: List[ExtractedLink] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    depth: int = 0
    parent_url: Optional[str] = None


@dataclass
class ResearchResult:
    """Complete research result with network analysis"""
    pdf_path: str
    total_links_found: int
    successful_scrapes: int
    failed_scrapes: int
    max_depth_reached: int
    content_network: nx.DiGraph
    scraped_content: List[ScrapedContent]
    summary: str = ""
    key_insights: List[str] = field(default_factory=list)
    domain_analysis: Dict[str, int] = field(default_factory=dict)


class WebScraperAgent:
    """Advanced web scraping with multiple fallback methods"""
    
    def __init__(self, use_selenium: bool = False, delay: float = 1.0):
        self.use_selenium = use_selenium
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.driver = None
        
    def __enter__(self):
        if self.use_selenium:
            self._setup_selenium()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.quit()
            
    def _setup_selenium(self):
        """Setup Selenium WebDriver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            self.driver = webdriver.Chrome(
                service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
                options=chrome_options
            )
        except Exception as e:
            logger.warning(f"Failed to setup Selenium: {e}")
            self.use_selenium = False
            
    def scrape_content(self, url: str, include_images: bool = True) -> ScrapedContent:
        """Scrape content from URL using multiple methods"""
        content = ScrapedContent(url=url)
        
        # Method 1: trafilatura (best for main content)
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                extracted = trafilatura.extract(
                    downloaded, 
                    include_images=include_images,
                    include_links=True,
                    include_formatting=True
                )
                if extracted:
                    content.content = extracted
                    content.success = True
                    
                    # Extract metadata
                    metadata = trafilatura.extract_metadata(downloaded)
                    if metadata:
                        content.title = metadata.title or ""
                        content.metadata = {
                            'author': metadata.author,
                            'date': metadata.date,
                            'description': metadata.description,
                            'sitename': metadata.sitename
                        }
        except Exception as e:
            logger.debug(f"Trafilatura failed for {url}: {e}")
            
        # Method 2: newspaper3k (good for articles)
        if not content.success:
            try:
                article = Article(url)
                article.download()
                article.parse()
                
                content.title = article.title
                content.content = article.text
                content.metadata = {
                    'authors': article.authors,
                    'publish_date': str(article.publish_date) if article.publish_date else None,
                    'top_image': article.top_image,
                    'keywords': article.keywords
                }
                if include_images and article.images:
                    content.images = list(article.images)
                content.success = True
            except Exception as e:
                logger.debug(f"Newspaper3k failed for {url}: {e}")
                
        # Method 3: Beautiful Soup (fallback)
        if not content.success:
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                content.title = title_tag.text.strip() if title_tag else ""
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                    
                # Extract main content
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                if main_content:
                    content.content = main_content.get_text(strip=True, separator='\n')
                else:
                    content.content = soup.get_text(strip=True, separator='\n')
                    
                # Extract images
                if include_images:
                    images = soup.find_all('img', src=True)
                    content.images = [urljoin(url, img['src']) for img in images[:10]]
                    
                content.success = True
            except Exception as e:
                content.error = str(e)
                logger.debug(f"BeautifulSoup failed for {url}: {e}")
                
        # Method 4: Selenium (for JavaScript-heavy sites)
        if not content.success and self.use_selenium and self.driver:
            try:
                self.driver.get(url)
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                content.title = self.driver.title
                content.content = self.driver.find_element(By.TAG_NAME, "body").text
                content.success = True
            except Exception as e:
                content.error = str(e)
                logger.debug(f"Selenium failed for {url}: {e}")
                
        # Extract links from content
        if content.success:
            content.links = self._extract_links_from_content(url, content.content)
            
        time.sleep(self.delay)
        return content
        
    def _extract_links_from_content(self, base_url: str, content: str) -> List[ExtractedLink]:
        """Extract links from scraped content"""
        links = []
        
        # Find URLs in text
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?]'
        found_urls = re.findall(url_pattern, content)
        
        for url in found_urls:
            if validators.url(url):
                links.append(ExtractedLink(
                    url=url,
                    text="",
                    context=self._extract_context(content, url),
                    parent_url=base_url
                ))
                
        return links
        
    def _extract_context(self, content: str, url: str, context_length: int = 100) -> str:
        """Extract context around a URL in content"""
        try:
            index = content.find(url)
            if index == -1:
                return ""
            
            start = max(0, index - context_length)
            end = min(len(content), index + len(url) + context_length)
            
            return content[start:end].strip()
        except Exception:
            return ""


class DeepResearcherAgent:
    """Main Deep Researcher Agent for PDF link extraction and web content analysis"""
    
    def __init__(self, 
                 use_selenium: bool = False,
                 delay_between_requests: float = 1.0,
                 max_workers: int = 5):
        self.use_selenium = use_selenium
        self.delay = delay_between_requests
        self.max_workers = max_workers
        self.scraped_urls: Set[str] = set()
        
    def extract_links_from_pdf(self, pdf_path: str) -> List[ExtractedLink]:
        """Extract all links from a PDF document"""
        links = []
        
        try:
            # Method 1: PyMuPDF (best for link extraction)
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract annotated links
                for link in page.get_links():
                    if link.get('uri'):
                        rect = link.get('from', fitz.Rect())
                        links.append(ExtractedLink(
                            url=link['uri'],
                            page_number=page_num + 1,
                            bbox=(rect.x0, rect.y0, rect.x1, rect.y1) if rect else None,
                            text=link.get('text', ''),
                            context=self._extract_pdf_context(page, rect) if rect else ""
                        ))
                
                # Extract URLs from text
                text = page.get_text()
                url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?]'
                found_urls = re.findall(url_pattern, text)
                
                for url in found_urls:
                    if validators.url(url):
                        links.append(ExtractedLink(
                            url=url,
                            page_number=page_num + 1,
                            context=self._extract_context_from_text(text, url)
                        ))
            
            doc.close()
            
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}, trying PyPDF2")
            
            # Fallback: PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;:!?]'
                        found_urls = re.findall(url_pattern, text)
                        
                        for url in found_urls:
                            if validators.url(url):
                                links.append(ExtractedLink(
                                    url=url,
                                    page_number=page_num + 1,
                                    context=self._extract_context_from_text(text, url)
                                ))
                                
            except Exception as e:
                logger.error(f"Failed to extract links from PDF: {e}")
        
        # Remove duplicates
        unique_links = {}
        for link in links:
            if link.url not in unique_links:
                unique_links[link.url] = link
                
        return list(unique_links.values())
    
    def _extract_pdf_context(self, page, rect, context_length: int = 200) -> str:
        """Extract context around a link in PDF"""
        try:
            # Get text blocks near the link
            blocks = page.get_text("dict")["blocks"]
            context = ""
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_rect = fitz.Rect(line["bbox"])
                        if line_rect.intersects(rect):
                            for span in line["spans"]:
                                context += span["text"] + " "
                                
            return context.strip()[:context_length]
        except Exception:
            return ""
    
    def _extract_context_from_text(self, text: str, url: str, context_length: int = 200) -> str:
        """Extract context around URL in text"""
        try:
            index = text.find(url)
            if index == -1:
                return ""
            
            start = max(0, index - context_length // 2)
            end = min(len(text), index + len(url) + context_length // 2)
            
            return text[start:end].strip()
        except Exception:
            return ""
    
    def filter_links(self, 
                    links: List[ExtractedLink],
                    allowed_domains: Optional[List[str]] = None,
                    blocked_domains: Optional[List[str]] = None,
                    relevance_threshold: float = 0.5) -> List[ExtractedLink]:
        """Filter links based on domain and relevance criteria"""
        filtered_links = []
        
        for link in links:
            try:
                parsed_url = urlparse(link.url)
                domain = parsed_url.netloc.lower()
                
                # Domain filtering
                if allowed_domains and not any(allowed_domain.lower() in domain 
                                             for allowed_domain in allowed_domains):
                    continue
                    
                if blocked_domains and any(blocked_domain.lower() in domain 
                                         for blocked_domain in blocked_domains):
                    continue
                
                # Basic relevance filtering
                if self._calculate_link_relevance(link) >= relevance_threshold:
                    filtered_links.append(link)
                    
            except Exception as e:
                logger.debug(f"Error filtering link {link.url}: {e}")
                
        return filtered_links
    
    def _calculate_link_relevance(self, link: ExtractedLink) -> float:
        """Calculate relevance score for a link"""
        score = 0.5  # Base score
        
        # Academic domains get higher score
        academic_domains = ['arxiv.org', 'scholar.google', 'pubmed', 'ieee.org', 
                          'acm.org', 'springer.com', 'nature.com', 'science.org']
        parsed_url = urlparse(link.url)
        domain = parsed_url.netloc.lower()
        
        if any(academic_domain in domain for academic_domain in academic_domains):
            score += 0.3
            
        # PDF links get higher score
        if link.url.lower().endswith('.pdf'):
            score += 0.2
            
        # Links with meaningful context get higher score
        if len(link.context) > 50:
            score += 0.1
            
        return min(score, 1.0)
    
    def scrape_links(self, 
                    links: List[ExtractedLink],
                    include_images: bool = True,
                    max_content_length: int = 10000) -> List[ScrapedContent]:
        """Scrape content from multiple links concurrently"""
        scraped_content = []
        
        with WebScraperAgent(use_selenium=self.use_selenium, delay=self.delay) as scraper:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit scraping tasks
                future_to_link = {
                    executor.submit(scraper.scrape_content, link.url, include_images): link 
                    for link in links
                }
                
                # Collect results with progress bar
                for future in tqdm(as_completed(future_to_link), 
                                 total=len(future_to_link), 
                                 desc="Scraping content"):
                    link = future_to_link[future]
                    try:
                        content = future.result(timeout=30)
                        content.depth = link.depth
                        content.parent_url = link.parent_url
                        
                        # Limit content length
                        if len(content.content) > max_content_length:
                            content.content = content.content[:max_content_length] + "..."
                            
                        scraped_content.append(content)
                        self.scraped_urls.add(content.url)
                        
                    except Exception as e:
                        logger.error(f"Failed to scrape {link.url}: {e}")
                        failed_content = ScrapedContent(
                            url=link.url,
                            success=False,
                            error=str(e),
                            depth=link.depth,
                            parent_url=link.parent_url
                        )
                        scraped_content.append(failed_content)
                        
        return scraped_content
    
    def multi_level_scraping(self,
                           initial_links: List[ExtractedLink],
                           max_depth: int = 2,
                           max_links_per_level: int = 10,
                           max_total_links: int = 50,
                           **scrape_kwargs) -> List[ScrapedContent]:
        """Perform multi-level recursive scraping"""
        all_scraped_content = []
        current_links = initial_links[:max_links_per_level]
        
        for depth in range(max_depth):
            if not current_links or len(all_scraped_content) >= max_total_links:
                break
                
            logger.info(f"Scraping depth {depth + 1} with {len(current_links)} links")
            
            # Set depth for current links
            for link in current_links:
                link.depth = depth
                
            # Scrape current level
            scraped_content = self.scrape_links(current_links, **scrape_kwargs)
            all_scraped_content.extend(scraped_content)
            
            # Collect new links for next level
            next_level_links = []
            for content in scraped_content:
                if content.success and content.links:
                    for link in content.links:
                        if (link.url not in self.scraped_urls and 
                            len(next_level_links) < max_links_per_level):
                            link.depth = depth + 1
                            link.parent_url = content.url
                            next_level_links.append(link)
                            
            current_links = next_level_links
            
        return all_scraped_content
    
    def build_content_network(self, scraped_content: List[ScrapedContent]) -> nx.DiGraph:
        """Build a network graph of scraped content and links"""
        G = nx.DiGraph()
        
        # Add nodes for each scraped content
        for content in scraped_content:
            G.add_node(content.url, 
                      title=content.title,
                      success=content.success,
                      depth=content.depth,
                      content_length=len(content.content),
                      num_images=len(content.images))
        
        # Add edges for parent-child relationships
        for content in scraped_content:
            if content.parent_url and content.parent_url in G:
                G.add_edge(content.parent_url, content.url)
                
        return G
    
    def analyze_domain_distribution(self, scraped_content: List[ScrapedContent]) -> Dict[str, int]:
        """Analyze distribution of domains in scraped content"""
        domain_counts = {}
        
        for content in scraped_content:
            try:
                domain = urlparse(content.url).netloc
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            except Exception:
                continue
                
        return dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True))
    
    def generate_summary(self, result: ResearchResult) -> str:
        """Generate a comprehensive summary of research findings"""
        summary_parts = []
        
        # Overview
        summary_parts.append(f"# Deep Research Analysis Report")
        summary_parts.append(f"**PDF Source:** {result.pdf_path}")
        summary_parts.append(f"**Total Links Found:** {result.total_links_found}")
        summary_parts.append(f"**Successful Scrapes:** {result.successful_scrapes}")
        summary_parts.append(f"**Failed Scrapes:** {result.failed_scrapes}")
        summary_parts.append(f"**Maximum Depth Reached:** {result.max_depth_reached}")
        summary_parts.append("")
        
        # Network analysis
        if result.content_network:
            summary_parts.append("## Network Analysis")
            summary_parts.append(f"- **Total Nodes:** {result.content_network.number_of_nodes()}")
            summary_parts.append(f"- **Total Edges:** {result.content_network.number_of_edges()}")
            
            if result.content_network.number_of_nodes() > 0:
                avg_degree = sum(dict(result.content_network.degree()).values()) / result.content_network.number_of_nodes()
                summary_parts.append(f"- **Average Degree:** {avg_degree:.2f}")
            summary_parts.append("")
        
        # Domain analysis
        if result.domain_analysis:
            summary_parts.append("## Domain Distribution")
            for domain, count in list(result.domain_analysis.items())[:10]:
                summary_parts.append(f"- **{domain}:** {count} pages")
            summary_parts.append("")
        
        # Content quality analysis
        successful_content = [c for c in result.scraped_content if c.success]
        if successful_content:
            summary_parts.append("## Content Quality Analysis")
            
            total_content_length = sum(len(c.content) for c in successful_content)
            avg_content_length = total_content_length / len(successful_content)
            summary_parts.append(f"- **Average Content Length:** {avg_content_length:.0f} characters")
            
            total_images = sum(len(c.images) for c in successful_content)
            summary_parts.append(f"- **Total Images Found:** {total_images}")
            
            # Readability analysis (if content exists)
            if successful_content and successful_content[0].content:
                try:
                    sample_content = successful_content[0].content
                    flesch_score = textstat.flesch_reading_ease(sample_content)
                    summary_parts.append(f"- **Sample Readability Score:** {flesch_score:.1f}")
                except Exception:
                    pass
            
            summary_parts.append("")
        
        # Key insights
        if result.key_insights:
            summary_parts.append("## Key Insights")
            for insight in result.key_insights:
                summary_parts.append(f"- {insight}")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def deep_research(self,
                     pdf_path: str,
                     max_links: int = 20,
                     allowed_domains: Optional[List[str]] = None,
                     blocked_domains: Optional[List[str]] = None,
                     include_images: bool = True,
                     max_depth: int = 2,
                     max_links_per_level: int = 10,
                     use_multi_level: bool = True,
                     **kwargs) -> ResearchResult:
        """Perform complete deep research workflow"""
        
        logger.info(f"Starting deep research on {pdf_path}")
        
        # Step 1: Extract links from PDF
        logger.info("Extracting links from PDF...")
        pdf_links = self.extract_links_from_pdf(pdf_path)
        logger.info(f"Found {len(pdf_links)} links in PDF")
        
        # Step 2: Filter links
        if allowed_domains or blocked_domains:
            logger.info("Filtering links...")
            pdf_links = self.filter_links(pdf_links, allowed_domains, blocked_domains)
            logger.info(f"After filtering: {len(pdf_links)} links")
        
        # Step 3: Limit number of links
        pdf_links = pdf_links[:max_links]
        
        # Step 4: Scrape content
        if use_multi_level:
            logger.info("Starting multi-level scraping...")
            scraped_content = self.multi_level_scraping(
                pdf_links,
                max_depth=max_depth,
                max_links_per_level=max_links_per_level,
                include_images=include_images,
                **kwargs
            )
        else:
            logger.info("Starting single-level scraping...")
            scraped_content = self.scrape_links(pdf_links, include_images=include_images)
        
        # Step 5: Build network and analyze
        content_network = self.build_content_network(scraped_content)
        domain_analysis = self.analyze_domain_distribution(scraped_content)
        
        # Step 6: Calculate statistics
        successful_scrapes = sum(1 for c in scraped_content if c.success)
        failed_scrapes = len(scraped_content) - successful_scrapes
        max_depth_reached = max((c.depth for c in scraped_content), default=0)
        
        # Step 7: Generate insights
        key_insights = self._generate_key_insights(scraped_content, content_network, domain_analysis)
        
        # Step 8: Create result
        result = ResearchResult(
            pdf_path=pdf_path,
            total_links_found=len(pdf_links),
            successful_scrapes=successful_scrapes,
            failed_scrapes=failed_scrapes,
            max_depth_reached=max_depth_reached,
            content_network=content_network,
            scraped_content=scraped_content,
            domain_analysis=domain_analysis,
            key_insights=key_insights
        )
        
        # Step 9: Generate summary
        result.summary = self.generate_summary(result)
        
        logger.info("Deep research completed successfully")
        return result
    
    def _generate_key_insights(self, 
                              scraped_content: List[ScrapedContent],
                              network: nx.DiGraph,
                              domain_analysis: Dict[str, int]) -> List[str]:
        """Generate key insights from the research"""
        insights = []
        
        successful_content = [c for c in scraped_content if c.success]
        
        # Content insights
        if successful_content:
            avg_content_length = sum(len(c.content) for c in successful_content) / len(successful_content)
            insights.append(f"Average content length is {avg_content_length:.0f} characters")
            
            images_count = sum(len(c.images) for c in successful_content)
            insights.append(f"Found {images_count} images across all scraped pages")
        
        # Network insights
        if network.number_of_nodes() > 1:
            density = nx.density(network)
            insights.append(f"Content network density is {density:.3f}")
            
            if nx.is_weakly_connected(network):
                insights.append("All content forms a connected network")
            else:
                components = list(nx.weakly_connected_components(network))
                insights.append(f"Content forms {len(components)} separate components")
        
        # Domain insights
        if domain_analysis:
            top_domain = list(domain_analysis.keys())[0]
            top_count = domain_analysis[top_domain]
            insights.append(f"Most referenced domain is {top_domain} with {top_count} pages")
            
            academic_domains = ['arxiv.org', 'scholar.google', 'pubmed', 'ieee.org', 'acm.org']
            academic_count = sum(count for domain, count in domain_analysis.items() 
                               if any(acad in domain for acad in academic_domains))
            if academic_count > 0:
                insights.append(f"Found {academic_count} academic sources")
        
        return insights