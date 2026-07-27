package com.novacycle.data.remote

import okhttp3.Interceptor
import okhttp3.Response
import java.io.IOException

/**
 * OkHttp interceptor that transparently retries failed requests.
 *
 * Retries on:
 *  - IOException (connection reset, socket timeout inside a single attempt, etc.)
 *  - HTTP 5xx server errors
 *
 * Never retries 4xx responses (client errors are deterministic).
 *
 * Backoff between attempts grows linearly: [backoffMillis] * attemptNumber.
 * With the default 2 retries and 500 ms base, a failing call costs at most
 * ~1.5 s of extra waiting on top of the request time itself.
 */
class RetryInterceptor(
    private val maxRetries: Int = 2,
    private val backoffMillis: Long = 500L,
    private val sleeper: (Long) -> Unit = { Thread.sleep(it) }
) : Interceptor {

    override fun intercept(chain: Interceptor.Chain): Response {
        var lastException: IOException? = null
        var lastResponse: Response? = null

        for (attempt in 0..maxRetries) {
            if (attempt > 0) {
                sleeper(backoffMillis * attempt)
                lastResponse?.close()
            }
            try {
                val response = chain.proceed(chain.request())
                if (response.code < 500) {
                    return response
                }
                // 5xx — retry unless this was the last attempt
                lastResponse = response
                lastException = null
            } catch (e: IOException) {
                lastException = e
                lastResponse = null
            }
        }

        lastResponse?.let { return it }
        throw lastException ?: IOException("Request failed after ${maxRetries + 1} attempts")
    }
}
