import re
import time
import json
import math
from pathlib import Path
from typing import Iterable, Set, List, Tuple, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://www.dohamagazine.qa/"
OUTPUT_TXT = "doha_magazine_articles.txt"
OUTPUT_JSON = "doha_magazine_articles.json"  # optional, helps you reuse the data
PROGRESS_JSON = "scraper_progress.json"  # save progress to resume if interrupted
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_REQUESTS = 1.0  # seconds (be polite!)
MAX_PAGES_PER_CATEGORY = None  # Set to a number (e.g., 5) for testing, None for all pages

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DM-Scraper/1.0; +https://example.org/omni)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Arabic category paths taken from the site menu
CATEGORY_PATHS = [
    "تقارير-وقضايا/",
    "حوارات/",
    "ملفات/",
    "مقالات/",
    "أدب/",
    "ترجمات/",
    "زوايا/",
]

# Also scrape the archive (can be slow - set to False for quick testing)
SCRAPE_ARCHIVE = False  # Set to True to scrape historical archive

READ_MORE_TEXTS = {"اقرا المزيد", "اقرأ المزيد", "اقرأ المَزيد", "اقرأ المـزيد"}  # be tolerant

def get(url: str, retries: int = 3) -> Optional[requests.Response]:
    """Fetch a URL with retry logic."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200 and r.text:
                return r
            elif r.status_code == 429:  
                wait = (attempt + 1) * 2
                print(f"    [!] Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            return None
        except requests.exceptions.Timeout:
            print(f"    [!] Timeout on attempt {attempt + 1}/{retries}: {url}")
            if attempt < retries - 1:
                time.sleep(2)
        except requests.RequestException as e:
            print(f"    [!] Error on attempt {attempt + 1}/{retries}: {str(e)[:50]}")
            if attempt < retries - 1:
                time.sleep(2)
    return None

def text_summary(full_text: str, max_words: int = 70) -> str:
    # Collapse whitespace
    t = re.sub(r"\s+", " ", full_text).strip()
    # Split into sentences (rough Arabic/Latin)
    # We’ll cut roughly by punctuation or word count
    sentences = re.split(r"(?<=[\.!\؟\!])\s+", t)
    if sentences and len(sentences[0].split()) >= max_words:
        # Very long first sentence → trim by words
        return " ".join(sentences[0].split()[:max_words]) + "…"
    out = []
    for s in sentences:
        if not s: 
            continue
        out.append(s)
        if len(" ".join(out).split()) >= max_words:
            break
    short = " ".join(out).strip()
    if len(short.split()) > max_words:
        short = " ".join(short.split()[:max_words]) + "…"
    return short

def parse_article_page(article_url: str) -> Optional[Tuple[str, str, str]]:
    """Return (title, url, summary) by opening the article page."""
    resp = get(article_url)
    if not resp:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")

    # Title: try common WP patterns then fallback to first h1
    title = None
    for sel in ["h1.entry-title", "header.entry-header h1", "article h1", "h1"]:
        node = soup.select_one(sel)
        if node and node.get_text(strip=True):
            title = node.get_text(strip=True)
            break
    if not title:
        # Fallback: page <title>
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        else:
            return None

    # Main content: typical WP selectors
    content_candidates = soup.select(".entry-content, article .entry-content, article, .post, .content")
    text = ""
    for node in content_candidates:
        # Avoid navigation/sidebar/footer noise
        if node.name in {"article", "div", "main", "section"}:
            txt = node.get_text(" ", strip=True)
            if txt and len(txt) > len(text):
                text = txt
    if not text:
        text = soup.get_text(" ", strip=True)  # ultimate fallback

    summary = text_summary(text, max_words=70)
    return (title, article_url, summary)

def extract_read_more_links(category_html: str, category_url: str) -> List[str]:
    soup = BeautifulSoup(category_html, "html.parser")
    links = []
    for a in soup.find_all("a"):
        label = (a.get_text() or "").strip().replace("\xa0", " ")
        if any(k in label for k in READ_MORE_TEXTS):
            href = a.get("href")
            if href:
                links.append(urljoin(category_url, href))
    # Remove non-article garbage or duplicates
    clean = []
    seen = set()
    for u in links:
        if u in seen: 
            continue
        seen.add(u)
        # crude filter: keep only same host
        if urlparse(u).netloc.endswith("dohamagazine.qa"):
            clean.append(u)
    return clean

def crawl_category(category_path: str) -> Iterable[str]:
    """Yield article URLs from a category across pages."""
    page = 1
    prev_links = set()  # Track previously seen links to detect duplicates
    
    while True:
        if MAX_PAGES_PER_CATEGORY and page > MAX_PAGES_PER_CATEGORY:
            print(f"      [i] Reached max pages limit ({MAX_PAGES_PER_CATEGORY})")
            break
        
        url = urljoin(BASE, category_path if page == 1 else f"{category_path}page/{page}/")
        print(f"      [i] Fetching page {page}: {url}")
        resp = get(url)
        if not resp:
            print(f"      [!] Failed to fetch page {page}")
            break
        
        article_links = extract_read_more_links(resp.text, url)
        
        # Stop if no new articles or same ones repeated (ghost pages)
        new_links = [u for u in article_links if u not in prev_links]
        if not new_links:
            print(f"      [!] No new articles found on page {page}. Stopping crawl for {category_path}")
            break
        
        print(f"      [+] Found {len(new_links)} new articles on page {page} (total on page: {len(article_links)})")
        for u in new_links:
            yield u
        
        prev_links.update(new_links)
        page += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)

def try_sitemap() -> List[str]:
    """Try to read the news sitemap for direct post URLs (fast path)."""
    sitemap_url = urljoin(BASE, "sitemap-news.xml")
    resp = get(sitemap_url)
    if not resp:
        return []
    soup = BeautifulSoup(resp.text, "xml")
    urls = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
    # Keep only article-looking paths (exclude PDFs, images, etc.)
    urls = [u for u in urls if urlparse(u).netloc.endswith("dohamagazine.qa")]
    return urls

def scrape_archive_page() -> List[str]:
    """Scrape the magazine archive page to get all historical issue links."""
    archive_url = urljoin(BASE, "أرشيف-المجلة/")
    resp = get(archive_url)
    if not resp:
        return []
    
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    
    # Find all links in the archive page
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        text = a.get_text(strip=True)
        
        # Look for issue links (they contain year/month patterns)
        if href and urlparse(href).netloc.endswith("dohamagazine.qa"):
            # Check if it looks like an issue page (contains numbers for dates)
            if any(char.isdigit() for char in text):
                full_url = urljoin(BASE, href)
                if full_url not in links:
                    links.append(full_url)
    
    return links

def scrape_homepage_articles() -> List[str]:
    """Scrape featured articles from the homepage."""
    resp = get(BASE)
    if not resp:
        return []
    
    return extract_read_more_links(resp.text, BASE)

def save_progress(collected: List[dict], processed_urls: Set[str]):
    """Save current progress to resume later if interrupted."""
    progress = {
        "collected": collected,
        "processed_urls": list(processed_urls),
        "timestamp": time.time()
    }
    with Path(PROGRESS_JSON).open("w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def load_progress() -> Tuple[List[dict], Set[str]]:
    """Load progress from previous interrupted run."""
    progress_file = Path(PROGRESS_JSON)
    if not progress_file.exists():
        return [], set()
    
    try:
        with progress_file.open("r", encoding="utf-8") as f:
            progress = json.load(f)
            return progress.get("collected", []), set(progress.get("processed_urls", []))
    except:
        return [], set()

def main():
    out_txt = Path(OUTPUT_TXT)
    out_json = Path(OUTPUT_JSON)

    # Try to load previous progress
    print("[*] Checking for previous progress...")
    collected, processed_urls = load_progress()
    if collected:
        print(f"[*] Resuming from previous run: {len(collected)} articles already collected")
    
    seen: Set[str] = processed_urls.copy()
    urls = []

    # 1) Fast path via sitemap (if available)
    print("[*] Trying sitemap...")
    sitemap_urls = try_sitemap()
    if sitemap_urls:
        print(f"[*] Found {len(sitemap_urls)} URLs from sitemap")
        urls.extend(sitemap_urls)

    # 2) Scrape homepage featured articles
    print("[*] Scraping homepage articles...")
    homepage_urls = scrape_homepage_articles()
    if homepage_urls:
        print(f"[*] Found {len(homepage_urls)} URLs from homepage")
        urls.extend(homepage_urls)

    # 3) Crawl all category pages
    print("[*] Crawling category pages...")
    for cat in CATEGORY_PATHS:
        print(f"    - Crawling category: {cat}")
        for u in crawl_category(cat):
            urls.append(u)

    # 4) Scrape archive if enabled
    if SCRAPE_ARCHIVE:
        print("[*] Scraping magazine archive...")
        archive_urls = scrape_archive_page()
        if archive_urls:
            print(f"[*] Found {len(archive_urls)} URLs from archive")
            # For each archive issue page, extract article links
            for issue_url in archive_urls:
                resp = get(issue_url)
                if resp:
                    issue_links = extract_read_more_links(resp.text, issue_url)
                    urls.extend(issue_links)
                    time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Deduplicate while preserving order
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    print(f"[*] Found {len(deduped)} new article URLs to process")
    print(f"[*] Total articles (including previous): {len(collected) + len(deduped)}")

    # Process articles
    for i, u in enumerate(deduped, 1):
        art = parse_article_page(u)
        if not art:
            print(f"[!] Skipped (failed): {u}")
            processed_urls.add(u)
            continue
        
        title, link, summary = art
        collected.append({"title": title, "link": link, "summary": summary})
        processed_urls.add(u)
        print(f"[{i}/{len(deduped)}] {title[:70]}")

        # Save progress every 10 articles
        if i % 10 == 0:
            save_progress(collected, processed_urls)

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Final save
    save_progress(collected, processed_urls)

    # Write TXT
    with out_txt.open("w", encoding="utf-8") as f:
        for item in collected:
            f.write(item["title"] + "\n")
            f.write(item["link"] + "\n")
            f.write(item["summary"] + "\n")
            f.write("-" * 80 + "\n")

    # Also write JSON (optional, handy for later reuse)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] Scraping complete!")
    print(f"[✓] Saved {len(collected)} articles to {out_txt.resolve()}")
    print(f"[✓] JSON copy: {out_json.resolve()}")
    
    # Clean up progress file
    progress_file = Path(PROGRESS_JSON)
    if progress_file.exists():
        progress_file.unlink()
        print(f"[✓] Cleaned up progress file")

if __name__ == "__main__":
    main()
