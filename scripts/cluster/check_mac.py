import urllib.request, json

mac = "50-2E-91"
try:
    req = urllib.request.urlopen(f"https://api.macvendors.com/{mac}", timeout=3)
    vendor = req.read().decode()
    print("MAC Vendor for 50-2E-91:", vendor)
except Exception as e:
    print("Vendor lookup failed:", e)
