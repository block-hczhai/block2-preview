
"""Build github pages for PYPI indexing."""

import sys, os, json, time
from urllib.request import Request, urlopen

REPO_URL = sys.argv[1]
DIST = sys.argv[2]

data = {}

headers = {
    "Authorization": "Bearer " + os.environ['GH_TOKEN'],
    "Content-Type": "application/json"
}

main_url = "https://api.github.com/repos/%s/releases" % REPO_URL

for d in json.loads(urlopen(Request(main_url + '?per_page=100', headers=headers)).read()):
    for dd in json.loads(urlopen(Request(d["assets_url"] + '?per_page=100', headers=headers)).read()):
        name = dd["name"]
        download_url = dd["browser_download_url"]
        if "-" not in name:
            continue
        package_name = name.split("-")[0]
        package_name = package_name.replace("_", "-")
        if package_name not in data:
            data[package_name] = {}
        data[package_name][name] = download_url
    time.sleep(0.5)

page = """<!DOCTYPE html>
<html>
  <body>
%s
  </body>
</html>
"""

if not os.path.exists(DIST):
    os.makedirs(DIST)

with open(DIST + "/index.html", "w") as f:
    cont = "\n".join("    <a href=\"%s/\">%s</a><br/>" % (x, x) for x in data)
    f.write(page % cont)

for k, v in sorted(data.items()):
    if not os.path.exists(DIST + "/" + k):
        os.makedirs(DIST + "/" + k)
    with open(DIST + "/" + k + "/index.html", "w") as f:
        cont = "    <h1>Links for %s</h1>\n" % k
        cont += "\n".join("    <a href=\"%s\">%s</a><br/>" % (y, x) for x, y in sorted(v.items()))
        f.write(page % cont)
