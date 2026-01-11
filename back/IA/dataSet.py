from playwright.sync_api import sync_playwright
from PIL import Image
from io import BytesIO
import psycopg2
import os
import time
import json
from dotenv import load_dotenv


load_dotenv()

DBHOST = os.getenv("dbHost")
DBNAME = os.getenv("dbName")
DBUSER = os.getenv("dbUser")
DBPORT = os.getenv("dbPort")
DBPASSWORD = os.getenv("dbPassword")
DBTABLE = os.getenv("dbTable")

URL = "https://www.pinterest.com/mandydv98/nature-pictures/"


conn = psycopg2.connect(
    host=DBHOST,
    database=DBNAME,
    user=DBUSER,
    password=DBPASSWORD,
    port=DBPORT
)
cur = conn.cursor()


img_urls = set()

def on_response(res):
    try:
        url = res.url
        if "i.pinimg.com" in url and res.status == 200:
            img_urls.add(url)
            print("[IMG]", url)
    except:
        pass


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(
        viewport={"width": 1280, "height": 900}
    )

    with open("cookies.json", "r") as f:
        context.add_cookies(json.load(f))

    page = context.new_page()
    page.on("response", on_response)

    page.goto(URL)
    page.wait_for_timeout(6000)

    for _ in range(40):
        page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
        time.sleep(1.2)

        # if len(img_urls) >= 200:
        #     break

    print(f"\n[+] images trouvées : {len(img_urls)}\n")

    page.remove_listener("response", on_response)


    for url in list(img_urls):  
        try:
            response = page.request.get(url, timeout=15000)
            img = Image.open(BytesIO(response.body())).convert("RGBA")

            buf_high = BytesIO()
            img.save(buf_high, "WEBP", quality=80)
            buf_high.seek(0)

            buf_low = BytesIO()
            img.save(buf_low, "WEBP", quality=20)
            buf_low.seek(0)

            cur.execute(
                f"""
                INSERT INTO {DBTABLE} (url, high, low)
                VALUES (%s, %s, %s)
                ON CONFLICT (url) DO NOTHING
                """,
                (
                    url,
                    psycopg2.Binary(buf_high.read()),
                    psycopg2.Binary(buf_low.read())
                )
            )

            conn.commit()
            print("[OK]", url)

        except Exception as e:
            print("[SKIP]", url, e)

    browser.close()

cur.close()
conn.close()
print("\nFIN")


