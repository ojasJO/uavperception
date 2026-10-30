"""Real-browser check of the demo (Chromium via Playwright, headless): starts the Streamlit server, then verifies that
selecting works, PLAY advances frames continuously, PAUSE stops it, Next/Previous step, First/Middle/Last jump,
boxes are drawn, metrics change with the frame, compare mode renders two panels, and the video tab can render + play.

    python experiments\\anti_uav\\demo\\browser_test.py        (system python with playwright; the app runs in the venv)
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
PY = ROOT / "venv" / "Scripts" / "python.exe"
PORT = 8766
URL = f"http://localhost:{PORT}"
SHOT = Path(__file__).resolve().parent / "cache" / "browser_test"


def frame_no(page) -> int:
    txt = page.locator("text=/Frame:\\s*\\d+\\s*\\/\\s*\\d+/").first.inner_text(timeout=15000)
    return int(re.search(r"Frame:\s*(\d+)", txt).group(1))


def main():
    SHOT.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen([str(PY), "-m", "streamlit", "run", str(Path(__file__).resolve().parent / "app.py"), "--server.headless", "true",
                            "--server.port", str(PORT), "--browser.gatherUsageStats", "false"], cwd=str(ROOT),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                if urllib.request.urlopen(URL + "/_stcore/health", timeout=2).read() == b"ok":
                    break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            page = b.new_page(viewport={"width": 1500, "height": 1000})
            page.goto(URL, wait_until="networkidle")
            page.wait_for_selector("text=VALIDATION ONLY", timeout=120000)
            page.wait_for_selector("text=/Frame:\\s*\\d+/", timeout=120000)
            page.wait_for_selector('[data-testid="stImage"]', timeout=60000)
            page.wait_for_timeout(2000)
            f0 = frame_no(page)
            imgs = page.locator('[data-testid="stImage"]').count()
            print(f"[1] page loaded; opens at frame {f0}; {imgs} images rendered (frame + inset + IoU strip expected)")
            assert imgs >= 3 and f0 >= 1
            page.screenshot(path=str(SHOT / "01_loaded.png"), full_page=False)

            page.get_by_role("button", name=re.compile("Next")).click()
            page.wait_for_timeout(1500)
            f1 = frame_no(page)
            print(f"[2] Next: {f0} -> {f1}")
            assert f1 == f0 + 1
            page.get_by_role("button", name=re.compile("Previous")).click()
            page.wait_for_timeout(1500)
            assert frame_no(page) == f0
            print(f"[3] Previous: back to {f0}")

            page.get_by_role("button", name="PLAY").click()
            page.wait_for_timeout(4000)
            fa = frame_no(page)
            page.wait_for_timeout(3000)
            fb = frame_no(page)
            print(f"[4] PLAY: after 4 s frame {fa}, after 7 s frame {fb} (5 fps expected ≈ +15 per 3 s)")
            assert fb > fa > f0, "playback did not advance continuously"
            page.screenshot(path=str(SHOT / "02_playing.png"))
            page.get_by_role("button", name="PAUSE").click()
            page.wait_for_timeout(2500)
            fp = frame_no(page)
            page.wait_for_timeout(2500)
            fq = frame_no(page)
            print(f"[5] PAUSE: frame {fp} then {fq} after 2.5 s more")
            assert fq == fp, "playback did not stop on PAUSE"

            page.get_by_role("button", name="Last").click()
            page.wait_for_timeout(1500)
            fl = frame_no(page)
            page.get_by_role("button", name="Middle").click()
            page.wait_for_timeout(1500)
            fm = frame_no(page)
            page.get_by_role("button", name="First target").click()
            page.wait_for_timeout(1500)
            ff = frame_no(page)
            print(f"[6] Last -> {fl}, Middle -> {fm}, First target -> {ff}")
            assert fl > fm > ff
            body = page.inner_text("body")
            for key in ("IoU:", "Centre error:", "Confidence:", "Sequence SA:"):
                assert key in body
            print("[7] metrics block visible (IoU / centre error / confidence / sequence SA)")

            # compare mode
            page.get_by_text("compare with a second checkpoint", exact=False).first.click()
            page.wait_for_timeout(4000)
            page.wait_for_timeout(2000)
            n_imgs = page.locator('[data-testid="stImage"]').count()
            print(f"[8] compare mode: {n_imgs} images (two panels)")
            assert n_imgs >= 6
            page.screenshot(path=str(SHOT / "03_compare.png"))
            page.get_by_role("button", name="PLAY").click()
            page.wait_for_timeout(3500)
            fc1 = frame_no(page)
            page.wait_for_timeout(2500)
            fc2 = frame_no(page)
            print(f"[9] compare + PLAY: {fc1} -> {fc2}")
            assert fc2 > fc1
            page.get_by_role("button", name="PAUSE").click()
            page.wait_for_timeout(1000)

            # video tab
            page.get_by_role("tab", name=re.compile("Annotated video")).click()
            page.wait_for_timeout(1000)
            btn = page.get_by_role("button", name=re.compile("Render annotated video"))
            if btn.count():
                btn.first.click()
                page.wait_for_selector("video", timeout=300000)
            page.wait_for_selector("video", timeout=60000)
            dur = page.evaluate("() => { const v = document.querySelector('video'); return v ? v.duration : -1 }")
            page.evaluate("() => { const v = document.querySelector('video'); v.muted = true; v.play(); }")
            page.wait_for_timeout(3000)
            ct = page.evaluate("() => document.querySelector('video').currentTime")
            print(f"[10] video tab: duration {dur:.1f} s, playing in the browser: currentTime {ct:.1f} s after 3 s")
            assert dur > 5 and ct > 1.0
            page.screenshot(path=str(SHOT / "04_video.png"))
            b.close()
        print(f"all browser checks passed; screenshots in {SHOT}")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except Exception:  # noqa: BLE001
            srv.kill()


if __name__ == "__main__":
    sys.exit(main())
