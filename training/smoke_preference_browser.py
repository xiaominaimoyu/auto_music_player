"""Exercise the actual file:// listening/export workflow on a synthetic bundle.

Optional developer tool: needs Playwright and an installed Chromium executable.
Refuses live collection bundles so automation can never generate human records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.preference_bundle import file_hash
from training.preference_dataset import validate_bundle, load_response


def smoke(*, bundle_root: Path, browser_executable: Path, output: Path) -> dict:
    from playwright.sync_api import sync_playwright

    bundle = validate_bundle(bundle_root)
    if not bundle["manifest"]["synthetic"]:
        raise ValueError("browser automation requires a synthetic bundle")
    if output.exists():
        raise ValueError("browser evidence output must be a new directory")
    output.mkdir(parents=True)
    (output / "responses").mkdir()
    (output / "_INCOMPLETE").touch()
    remote_requests, errors, downloads = [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(browser_executable), headless=True,
                                              args=["--disable-gpu"])
        try:
            context = browser.new_context(accept_downloads=True, viewport={"width": 1050, "height": 800})
            page = context.new_page()
            page.on("request", lambda request: remote_requests.append(request.url)
                    if request.url.startswith(("https:", "http:")) else None)
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("download", lambda download: downloads.append(download))
            page.goto((bundle_root / "public" / "index.html").resolve().as_uri())
            assert page.locator("h1").inner_text() == "离线演奏试听"
            assert not page.locator("#consent").is_checked()
            radios = page.locator("input[type=radio]")
            assert not any(radios.nth(i).is_checked() for i in range(radios.count()))
            assert not page.locator("input[value=a]").first.is_enabled()
            page.locator("#export").click()
            assert "未同意时不会导出" in page.locator("#status").inner_text()
            assert not downloads
            played = page.evaluate("""async () => {
                const result = [];
                for (const audio of document.querySelectorAll('audio')) {
                    audio.muted = true; audio.playbackRate = 16;
                    await new Promise((resolve,reject) => {
                        const timer=setTimeout(()=>reject(new Error('audio timeout')),8000);
                        audio.addEventListener('ended',()=>{clearTimeout(timer);resolve();},{once:true});
                        audio.play().catch(reject);
                    });
                    result.push({duration:audio.duration, ended:audio.ended, error:audio.error});
                }
                return result;
            }""")
            assert all(a["ended"] and a["error"] is None for a in played)
            assert page.locator("input[value=a]").first.is_enabled()
            sections = page.locator("#trials > section")
            choices = ("a", "tie", "skip")
            for index in range(sections.count()):
                sections.nth(index).locator(f"input[value={choices[index % 3]}]").check()
            page.locator("#consent").check()
            with page.expect_download() as pending:
                page.locator("#export").click()
            download = pending.value
            failure = download.failure()
            if failure is not None:
                raise RuntimeError(f"browser download failed: {failure}")
            response_path = output / "responses" / "browser_response.json"
            download.save_as(response_path)
            assert "已导出答卷" in page.locator("#status").inner_text()
            page.screenshot(path=str(output / "listening_export.png"), full_page=True)
            validated = load_response(response_path, bundle=bundle)
            if remote_requests or errors:
                raise ValueError(f"unexpected page errors/remote requests: {errors}, {remote_requests}")
            result = {"status": "complete", "synthetic": True, "human_preferences_created": 0,
                      "consent_default_false": True, "no_default_choice": True,
                      "no_consent_export_blocked": True, "actual_audio_played": played,
                      "response_sha256": file_hash(response_path), "suggested_filename": download.suggested_filename,
                      "remote_requests": remote_requests, "page_errors": errors,
                      "response_validation_passed": True,
                      "audio_playback_for_engineering_only_muted_accelerated": True}
        finally:
            browser.close()
    (output / "browser_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "_INCOMPLETE").unlink()
    (output / "_SUCCESS").write_text(file_hash(output / "browser_summary.json") + "\n", encoding="ascii")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle-root", "browser-executable", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    result = smoke(**vars(parser.parse_args(argv)))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
