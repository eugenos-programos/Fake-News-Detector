import trafilatura
 

class WebScraper():
    def __init__(self) -> None:
        pass

    def gather_text(self, url: str) -> str:
        html_text = trafilatura.fetch_url(url=url)
        if html_text is None:
            raise ValueError("Prolems with gathering URL text.")
        text = trafilatura.extract(html_text)
        if text is None or 'JavaScript' in text:
            raise ValueError("Prolems with gathering URL text.")
        return text
