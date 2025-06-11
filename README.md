# README: Testing `parse` Function

This README demonstrates how to test the features of the `parse` function, which processes Markdown content and extracts URLs and code snippets while replacing them with placeholders.

## Function Overview

The `parse` function performs the following tasks on a given Markdown content (`md_content`):

- Replaces inline links in the format `[text](url)` with placeholders `{{URL_<index>}}`.
- Replaces standalone URLs (e.g., `http://example.com`) with placeholders `{{URL_<index>}}`.
- Replaces inline code snippets (e.g., \`code\`) with placeholders `{{CODE_<index>}}`.
- Replaces fenced code blocks (e.g., \`\`\` code \`\`\`) with placeholders `{{CODE_<index>}}`.

The function returns the modified content along with two lists:
1. `urls` – A list of all the URLs found in the Markdown content.
2. `code` – A list of all the code snippets (inline or fenced) found in the content.

## Example Usage

```python
import re

def parse(md_content):

    urls = []
    code = []

    # Replace inline links [text](url)
    def url_replacer(match):
        urls.append(match.group(0))
        return f"{{URL_{len(urls)-1}}}"
    
    content = re.sub(r'\[.*?\]\(http[s]?://\S+\)', url_replacer, md_content)

    # Replace standalone URLs
    def standalone_url_replacer(match):
        urls.append(match.group(0))
        return f"{{URL_{len(urls)-1}}}"
    
    md_content = re.sub(r'http[s]?://\S+', standalone_url_replacer, md_content)

    # Replace inline code snippets `code`
    def inline_code_replacer(match):
        code.append(match.group(0))
        return f"{{CODE_{len(code)-1}}}"
    
    md_content = re.sub(r'`[^`]+`', inline_code_replacer, md_content)

    # Replace fenced code blocks ``` code ```
    def fenced_code_replacer(match):
        code.append(match.group(0))
        return f"{{CODE_{len(code)-1}}}"
    
    content = re.sub(r'```.*?```', fenced_code_replacer, content, flags=re.DOTALL)

    return content, urls, code
