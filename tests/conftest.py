"""Keep SDK 3's httpx2 requests inside pytest-httpx's mock dispatcher."""

import httpx
import httpx2
import pytest


class SyncMockStream(httpx2.SyncByteStream):
    def __init__(self, response):
        self.response = response

    def __iter__(self):
        yield from self.response.iter_raw()

    def close(self):
        self.response.close()


class AsyncMockStream(httpx2.AsyncByteStream):
    def __init__(self, response):
        self.response = response

    async def __aiter__(self):
        async for chunk in self.response.aiter_raw():
            yield chunk

    async def aclose(self):
        await self.response.aclose()


@pytest.fixture(autouse=True)
def mock_httpx2(monkeypatch, httpx_mock):
    # Call the mock dispatcher directly, never a real transport. This preserves
    # pytest-httpx's matching and request assertions without allowing live I/O.
    def handle_request(transport, request):
        converted = httpx.Request(
            request.method,
            str(request.url),
            headers=request.headers.raw,
            content=request.read(),
            extensions=request.extensions,
        )
        response = httpx_mock._handle_request(transport, converted)
        return httpx2.Response(
            response.status_code,
            headers=response.headers.raw,
            stream=SyncMockStream(response),
            extensions=response.extensions,
        )

    async def handle_async_request(transport, request):
        converted = httpx.Request(
            request.method,
            str(request.url),
            headers=request.headers.raw,
            content=await request.aread(),
            extensions=request.extensions,
        )
        response = await httpx_mock._handle_async_request(transport, converted)
        return httpx2.Response(
            response.status_code,
            headers=response.headers.raw,
            stream=AsyncMockStream(response),
            extensions=response.extensions,
        )

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", handle_request)
    monkeypatch.setattr(
        httpx2.AsyncHTTPTransport, "handle_async_request", handle_async_request
    )
