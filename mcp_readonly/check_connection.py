"""Verify the live HTTP MCP contract, not just the health endpoint."""
import asyncio
import json
import os
from mcp import Client


async def main():
    url = os.environ.get("SF02_CHECK_URL", "http://127.0.0.1:8765/mcp")
    async with Client(url, read_timeout_seconds=15) as client:
        listed = await client.list_tools()
        names = {t.name for t in listed.tools}
        assert names == {"search", "fetch", "get_project_status"}, names
        for tool in listed.tools:
            a = tool.annotations
            assert a and a.read_only_hint and not a.destructive_hint and not a.open_world_hint
        found = await client.call_tool("search", {"query": "当前进度和训练方向"})
        assert not found.is_error and found.structured_content["results"]
        status = await client.call_tool("get_project_status", {})
        assert not status.is_error
        fetched = await client.call_tool("fetch", {"id": status.structured_content["source_ids"][0]})
        assert not fetched.is_error and fetched.structured_content["text"]
        for value in (found, fetched, status):
            texts = [json.loads(c.text) for c in value.content if c.type == "text"]
            assert value.structured_content in texts, "JSON text differs from structuredContent"
        print(json.dumps({"local_mcp_passed": True, "url": url,
                          "tools": sorted(names), "project_as_of": status.structured_content["as_of"],
                          "experiment_status": status.structured_content["experiment_status"],
                          "chatgpt_connected": "not_verified_by_local_check"}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
