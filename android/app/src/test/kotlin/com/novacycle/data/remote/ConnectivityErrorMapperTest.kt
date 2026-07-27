package com.novacycle.data.remote

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import com.squareup.moshi.JsonDataException
import com.squareup.moshi.JsonEncodingException
import org.junit.Test
import retrofit2.HttpException
import retrofit2.Response
import java.io.IOException
import java.io.InterruptedIOException
import java.net.ConnectException
import java.net.NoRouteToHostException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import javax.net.ssl.SSLHandshakeException

/**
 * The classifier must map every plausible network failure to a specific code
 * with a human-readable message — the app must never show "null".
 */
class ConnectivityErrorMapperTest {

    private fun classify(t: Throwable) = ConnectivityErrorMapper.classify(t)

    @Test
    fun `unknown host maps to DNS_FAILURE`() {
        val err = classify(UnknownHostException("Unable to resolve host novacycle.example"))
        assertEquals(ConnectivityErrorCode.DNS_FAILURE, err.code)
    }

    @Test
    fun `socket timeout maps to TIMEOUT`() {
        val err = classify(SocketTimeoutException("timeout"))
        assertEquals(ConnectivityErrorCode.TIMEOUT, err.code)
    }

    @Test
    fun `okhttp call timeout (InterruptedIOException) maps to TIMEOUT`() {
        val err = classify(InterruptedIOException("timeout"))
        assertEquals(ConnectivityErrorCode.TIMEOUT, err.code)
    }

    @Test
    fun `connection refused maps to NETWORK_UNREACHABLE`() {
        val err = classify(ConnectException("Failed to connect to /192.168.1.25:8000"))
        assertEquals(ConnectivityErrorCode.NETWORK_UNREACHABLE, err.code)
    }

    @Test
    fun `no route to host maps to NETWORK_UNREACHABLE`() {
        val err = classify(NoRouteToHostException("No route to host"))
        assertEquals(ConnectivityErrorCode.NETWORK_UNREACHABLE, err.code)
    }

    @Test
    fun `ssl failure maps to SSL_FAILURE`() {
        val err = classify(SSLHandshakeException("Handshake failed"))
        assertEquals(ConnectivityErrorCode.SSL_FAILURE, err.code)
    }

    @Test
    fun `http 5xx maps to BACKEND_DOWN`() {
        val body = "".toResponseBody("application/json".toMediaType())
        val err = classify(HttpException(Response.error<Any>(503, body)))
        assertEquals(ConnectivityErrorCode.BACKEND_DOWN, err.code)
    }

    @Test
    fun `http 4xx maps to UNKNOWN not BACKEND_DOWN`() {
        val body = "".toResponseBody("application/json".toMediaType())
        val err = classify(HttpException(Response.error<Any>(404, body)))
        assertEquals(ConnectivityErrorCode.UNKNOWN, err.code)
    }

    @Test
    fun `generic io exception maps to NETWORK_UNREACHABLE`() {
        val err = classify(IOException("unexpected end of stream"))
        assertEquals(ConnectivityErrorCode.NETWORK_UNREACHABLE, err.code)
    }

    @Test
    fun `json parsing failure maps to BACKEND_RESPONSE_INVALID`() {
        val err = classify(JsonDataException("Expected string but was BEGIN_OBJECT"))
        assertEquals(ConnectivityErrorCode.BACKEND_RESPONSE_INVALID, err.code)
    }

    @Test
    fun `malformed json maps to BACKEND_RESPONSE_INVALID`() {
        val err = classify(JsonEncodingException("Use JsonReader.setLenient(true)"))
        assertEquals(ConnectivityErrorCode.BACKEND_RESPONSE_INVALID, err.code)
    }

    @Test
    fun `every code has a non-blank user message even with null exception message`() {
        // Regression for "Could not reach server: null"
        val err = classify(ConnectException()) // message == null
        assertNotNull(err.userMessage)
        assert(err.userMessage.isNotBlank())
        ConnectivityErrorCode.values().forEach { code ->
            assert(code.userMessage.isNotBlank())
        }
    }
}
