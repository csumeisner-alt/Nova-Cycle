package com.novacycle.data.remote

import okhttp3.Interceptor
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import java.io.IOException

/**
 * Verifies the bounded retry-with-backoff behavior of [RetryInterceptor]
 * using a fake OkHttp chain (no real network).
 */
class RetryInterceptorTest {

    private val request = Request.Builder().url("http://example.test/api/healthz").build()

    private fun response(code: Int): Response = Response.Builder()
        .request(request)
        .protocol(Protocol.HTTP_1_1)
        .code(code)
        .message("test")
        .body("".toResponseBody(null))
        .build()

    /** Fake chain that replays scripted outcomes: an Int (HTTP code) or an IOException. */
    private class FakeChain(
        private val outcomes: MutableList<Any>,
        private val request: Request
    ) : Interceptor.Chain {
        var calls = 0
        override fun request(): Request = request
        override fun proceed(request: Request): Response {
            calls++
            return when (val outcome = outcomes.removeAt(0)) {
                is Response -> outcome
                is IOException -> throw outcome
                else -> error("bad outcome")
            }
        }
        override fun connection() = null
        override fun call() = throw UnsupportedOperationException()
        override fun connectTimeoutMillis() = 0
        override fun readTimeoutMillis() = 0
        override fun writeTimeoutMillis() = 0
        override fun withConnectTimeout(timeout: Int, unit: java.util.concurrent.TimeUnit) = this
        override fun withReadTimeout(timeout: Int, unit: java.util.concurrent.TimeUnit) = this
        override fun withWriteTimeout(timeout: Int, unit: java.util.concurrent.TimeUnit) = this
    }

    private fun interceptor(sleeps: MutableList<Long> = mutableListOf()) =
        RetryInterceptor(maxRetries = 2, backoffMillis = 100L, sleeper = { sleeps.add(it) })

    @Test
    fun `successful response returns immediately without retries`() {
        val chain = FakeChain(mutableListOf(response(200)), request)
        val resp = interceptor().intercept(chain)
        assertEquals(200, resp.code)
        assertEquals(1, chain.calls)
    }

    @Test
    fun `io exception retried then succeeds`() {
        val chain = FakeChain(
            mutableListOf(IOException("reset"), response(200)), request
        )
        val resp = interceptor().intercept(chain)
        assertEquals(200, resp.code)
        assertEquals(2, chain.calls)
    }

    @Test
    fun `server 5xx retried then succeeds`() {
        val chain = FakeChain(
            mutableListOf(response(503), response(200)), request
        )
        val resp = interceptor().intercept(chain)
        assertEquals(200, resp.code)
        assertEquals(2, chain.calls)
    }

    @Test
    fun `client 4xx never retried`() {
        val chain = FakeChain(mutableListOf(response(404)), request)
        val resp = interceptor().intercept(chain)
        assertEquals(404, resp.code)
        assertEquals(1, chain.calls)
    }

    @Test
    fun `persistent io exception rethrown after max retries`() {
        val chain = FakeChain(
            mutableListOf(IOException("a"), IOException("b"), IOException("c")), request
        )
        assertThrows(IOException::class.java) { interceptor().intercept(chain) }
        assertEquals(3, chain.calls) // 1 original + 2 retries
    }

    @Test
    fun `persistent 5xx returns last response after max retries`() {
        val chain = FakeChain(
            mutableListOf(response(500), response(502), response(503)), request
        )
        val resp = interceptor().intercept(chain)
        assertEquals(503, resp.code)
        assertEquals(3, chain.calls)
    }

    @Test
    fun `backoff grows linearly between attempts`() {
        val sleeps = mutableListOf<Long>()
        val chain = FakeChain(
            mutableListOf(IOException("a"), IOException("b"), response(200)), request
        )
        interceptor(sleeps).intercept(chain)
        assertEquals(listOf(100L, 200L), sleeps)
    }
}
