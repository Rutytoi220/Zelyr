import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
from zelyr.tools.browser_extension import get_bridge
from zelyr.tools.browser_cdp import InteractWithBrowserTool

EXT_DIR = Path(__file__).resolve().parent / "zelyr" / "tools" / "extension"

async def main():
    print("[1/5] Starting Zelyr WebSocket daemon on 127.0.0.1:41144...")
    bridge = get_bridge()
    asyncio.create_task(bridge.start_server())
    await asyncio.sleep(0.3)

    print(f"[2/5] Launching Playwright Chromium with unpacked extension from {EXT_DIR}...")
    user_data_dir = Path("/tmp/playwright_zelyr_profile")
    if user_data_dir.exists():
        import shutil
        shutil.rmtree(user_data_dir, ignore_errors=True)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=False,
            args=[
                f"--disable-extensions-except={EXT_DIR}",
                f"--load-extension={EXT_DIR}",
                "--no-sandbox",
            ],
        )

        # Forward extension background service worker console logs directly to terminal stdout
        def on_service_worker(sw):
            print(f"  [Service Worker Active] {sw.url}")
            sw.on("console", lambda msg: print(f"  [Worker Log] {msg.text}"))
        context.on("serviceworker", on_service_worker)
        for sw in context.service_workers:
            on_service_worker(sw)

        print("[3/5] Waiting for extension to handshake with ws://127.0.0.1:41144...")
        for _ in range(30):
            if bridge.is_connected():
                break
            await asyncio.sleep(0.2)

        if not bridge.is_connected():
            print("❌ Extension failed to connect to daemon within 6 seconds.")
            await context.close()
            await bridge.stop_server()
            return

        print("✓ Extension connected to Zelyr daemon.")

        print("[4/5] Opening test tabs (DuckDuckGo & Gemini)...")
        page_ddg = await context.new_page()
        await page_ddg.goto("https://duckduckgo.com")

        page_gemini = await context.new_page()
        await page_gemini.goto("https://gemini.google.com")
        await asyncio.sleep(1.0)

        # Focus DuckDuckGo first so we test switching focus back to Gemini
        await page_ddg.bring_to_front()
        print("  Active tab before dispatch: DuckDuckGo")

        print("[5/5] Executing tool dispatch: switch_tab(query='gemini')...")
        tool = InteractWithBrowserTool()
        result = await tool.execute({"action": "switch_tab", "query": "gemini"})
        print(f"  Tool Execution Result: {result}")
        assert result.success is True, f"switch_tab failed: {result.error}"

        await asyncio.sleep(1.0)

        # Verify which page is frontmost
        active_title = await context.pages[-1].title() if context.pages else "Unknown"
        print(f"✓ Active page title: '{active_title}'")

        # -------------------------------------------------------------------
        # The Soak Phase: 35 seconds of idle wait past the 30-second MV3 cliff
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("[SOAK PHASE] Starting 35-second idle soak test past the MV3 30s cliff...")
        print("  Monitoring stdout for background heartbeat frames ('ping' / 'pong')...")
        print("=" * 60)

        soak_duration = 35
        for elapsed in range(5, soak_duration + 1, 5):
            await asyncio.sleep(5)
            print(f"  [Soak Telemetry] Elapsed {elapsed}s / {soak_duration}s (active clients: {bridge.client_count})")
            assert bridge.is_connected() is True, f"Connection died at {elapsed}s during soak!"

        print("✓ 35s soak phase completed successfully without connection drop.")
        assert bridge.is_connected() is True, "Extension disconnected after 35s soak phase!"

        # -------------------------------------------------------------------
        # Post-Soak Verification: Send second tool command past the 30s cliff
        # -------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("[POST-SOAK VERIFICATION] Executing second tool command: list_tabs()...")
        print("=" * 60)
        list_res = await tool.execute({"action": "list_tabs"})
        print(f"  Post-Soak Tool Result: {list_res}")
        assert list_res.success is True, f"Post-soak list_tabs failed: {list_res.error}"
        tabs = list_res.output if isinstance(list_res.output, list) else list_res.output.get("tabs", [])
        assert len(tabs) >= 2, f"Expected at least 2 open tabs, found {len(tabs)}"
        print(f"✓ Successfully retrieved {len(tabs)} tabs post-soak:")
        for t in tabs:
            print(f"    • [{t.get('id')}] {t.get('title')} ({t.get('url')})")

        await context.close()
        await bridge.stop_server()
        print("\n============================================================")
        print("✅ 35-SECOND LONGEVITY / SOAK TEST PASSED CLEANLY!")
        print("============================================================\n")

if __name__ == "__main__":
    asyncio.run(main())
