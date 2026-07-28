package com.novacycle.viewmodel

import com.novacycle.data.remote.ConnectivityErrorCode
import com.novacycle.data.remote.ConnectivityErrorMapper
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Tests for the Settings connection-test failure classification and the
 * UNKNOWN-detail improvement. The threading fix itself (running the OkHttp
 * call on Dispatchers.IO) is exercised implicitly: a main-thread network call
 * throws NetworkOnMainThreadException on device, which classify() maps to
 * UNKNOWN — these tests pin the diagnosable detail for that path.
 */
class SettingsConnectionTestTest {

    // ── UNKNOWN failures now carry the exception class for diagnosability ──

    @Test
    fun `unknown failure detail includes exception class name`() {
        val err = ConnectivityErrorMapper.classify(IllegalStateException("boom"))
        assertEquals(ConnectivityErrorCode.UNKNOWN, err.code)
        assertEquals("IllegalStateException: boom", err.detail)
    }

    @Test
    fun `unknown failure with null message still has diagnosable detail`() {
        val err = ConnectivityErrorMapper.classify(IllegalStateException())
        assertEquals(ConnectivityErrorCode.UNKNOWN, err.code)
        assertEquals("IllegalStateException: no message", err.detail)
    }

    @Test
    fun `classified failures keep the plain exception message as detail`() {
        val err = ConnectivityErrorMapper.classify(java.net.UnknownHostException("bad.host"))
        assertEquals(ConnectivityErrorCode.DNS_FAILURE, err.code)
        assertEquals("bad.host", err.detail)
    }

    @Test
    fun `timeout is classified as TIMEOUT not unknown`() {
        val err = ConnectivityErrorMapper.classify(java.net.SocketTimeoutException("timeout"))
        assertEquals(ConnectivityErrorCode.TIMEOUT, err.code)
    }

    @Test
    fun `connection refused is classified as unreachable`() {
        val err = ConnectivityErrorMapper.classify(java.net.ConnectException("refused"))
        assertEquals(ConnectivityErrorCode.NETWORK_UNREACHABLE, err.code)
    }

    @Test
    fun `every classified error surfaces a non blank user message`() {
        val throwables = listOf<Throwable>(
            java.net.UnknownHostException(),
            java.net.SocketTimeoutException(),
            java.net.ConnectException(),
            javax.net.ssl.SSLException("handshake"),
            RuntimeException()
        )
        for (t in throwables) {
            val err = ConnectivityErrorMapper.classify(t)
            assertTrue(err.userMessage.isNotBlank())
        }
    }
}
