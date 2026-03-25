import os
import urllib.request
import urllib.parse
import json
import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def fetch_wikimedia_images(query, count, output_dir, start_idx):
    endpoint = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": int(count),
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }
    
    query_string = urllib.parse.urlencode(params)
    url = f"{endpoint}?{query_string}"
    
    req = urllib.request.Request(url, headers={'User-Agent': 'HumanitarianLogisticsBot/1.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
        pages = data.get("query", {}).get("pages", {})
        idx = start_idx
        for page_id, page_data in pages.items():
            imageinfo = page_data.get("imageinfo", [])
            if not imageinfo: continue
            
            img_url = imageinfo[0].get("url")
            if not img_url: continue
            
            if not img_url.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            ext = img_url.split('.')[-1]
            filename = os.path.join(output_dir, f"wiki_{idx:03d}.{ext}")
            
            try:
                img_req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(img_req, timeout=10) as img_res, open(filename, 'wb') as f:
                    f.write(img_res.read())
                print(f"Downloaded {filename}")
                idx += 1
            except Exception as e:
                print(f"Failed to download {img_url}: {e}")
                
        return idx
    except Exception as e:
        print(f"API Error for query '{query}': {e}")
        return start_idx

def main():
    output_dir = "/Users/sadhammydeen/Documents/humanitarian_logistics/data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    queries = [
        "rice sack", 
        "cardboard box pile",
        "relief supplies",
        "clothing donations",
        "humanitarian aid bags"
    ]
    
    idx = 1
    for q in queries:
        print(f"Searching Wikimedia for: {q}")
        idx = fetch_wikimedia_images(q, 50, output_dir, idx)
        time.sleep(1) # respectful delay
        
    print(f"Dataset generation complete. Total downloaded: {idx - 1}")

if __name__ == "__main__":
    main()
