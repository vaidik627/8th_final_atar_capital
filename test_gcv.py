import urllib.request, urllib.error, json, base64

api_key = "AIzaSyABYToZ-B5aqpum1h7l0rai6OIfrvqtlFQ"

# Minimal 1x1 white PNG to test API connectivity only
png_1x1 = base64.b64encode(
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
    b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00'
    b'\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
).decode("utf-8")

body = json.dumps({
    "requests": [{
        "image": {"content": png_1x1},
        "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]
    }]
}).encode()

url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")

try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read().decode())
        print("SUCCESS — API key is valid and Vision API is enabled")
except urllib.error.HTTPError as e:
    err_body = e.read().decode()
    print(f"FAILED — HTTP {e.code}")
    print(err_body[:600])
except Exception as e:
    print(f"ERROR: {e}")
