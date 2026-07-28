package com.novacycle.data.remote

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import dagger.hilt.android.qualifiers.ApplicationContext
import java.net.ConnectException
import java.net.NoRouteToHostException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import com.squareup.moshi.JsonDataException
import com.squareup.moshi.JsonEncodingException
import javax.inject.Inject
import javax.inject.Singleton
import javax.net.ssl.SSLException

/**
 * Structured connectivity error codes shared by the Settings connection test,
 * the repository, and the dashboard error banner. Each maps to a specific,
 * human-readable message — the app must never surface a raw "null" message.
 */
enum class ConnectivityErrorCode(val userMessage: String) {
    NETWORK_OFFLINE("No internet connection — check Wi-Fi or mobile data"),
    DNS_FAILURE("Could not find the server — check the URL's domain name"),
    NETWORK_UNREACHABLE("Server unreachable — check the address, port, and that the backend is running"),
    TIMEOUT("Server took too long to respond — it may be starting up, try again"),
    SSL_FAILURE("Secure connection failed — check that the URL uses the right http:// or https:// scheme"),
    BACKEND_DOWN("Server responded with an error — the backend may be down or misconfigured"),
    BACKEND_RESPONSE_INVALID("Server response didn't match the app — the backend or app may be mismatched"),
    UNKNOWN("Connection failed for an unknown reason");
}

/** A classified connectivity failure with a code and a display-ready message. */
data class ConnectivityError(
    val code: ConnectivityErrorCode,
    val userMessage: String,
    val detail: String?
)

/** Checks whether the device currently has any validated network connection. */
@Singleton
class NetworkStatusChecker @Inject constructor(
    @ApplicationContext private val context: Context
) {
    fun isOffline(): Boolean {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE)
            as? ConnectivityManager ?: return false
        val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return true
        return !caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }
}

/**
 * Maps low-level exceptions from OkHttp/Retrofit into [ConnectivityError]s.
 *
 * Pure classification logic is in [classify] (unit-testable, no Android deps);
 * [map] additionally consults the device network state so that "device is
 * offline" wins over more specific socket errors.
 */
@Singleton
class ConnectivityErrorMapper @Inject constructor(
    private val networkStatusChecker: NetworkStatusChecker
) {

    fun map(throwable: Throwable): ConnectivityError {
        if (networkStatusChecker.isOffline()) {
            return ConnectivityError(
                code = ConnectivityErrorCode.NETWORK_OFFLINE,
                userMessage = ConnectivityErrorCode.NETWORK_OFFLINE.userMessage,
                detail = throwable.message
            )
        }
        return classify(throwable)
    }

    companion object {
        /** Exception → error code classification, independent of device state. */
        fun classify(throwable: Throwable): ConnectivityError {
            val code = when (throwable) {
                is UnknownHostException -> ConnectivityErrorCode.DNS_FAILURE
                is SocketTimeoutException -> ConnectivityErrorCode.TIMEOUT
                is java.io.InterruptedIOException ->
                    // OkHttp callTimeout throws InterruptedIOException("timeout")
                    ConnectivityErrorCode.TIMEOUT
                is ConnectException,
                is NoRouteToHostException -> ConnectivityErrorCode.NETWORK_UNREACHABLE
                is SSLException -> ConnectivityErrorCode.SSL_FAILURE
                is retrofit2.HttpException ->
                    when (throwable.code()) {
                        in 500..599 -> ConnectivityErrorCode.BACKEND_DOWN
                        404 -> ConnectivityErrorCode.UNKNOWN // endpoint not found — usually a wrong URL
                        in 400..499 -> ConnectivityErrorCode.UNKNOWN // client-side issue, but keep structured
                        else -> ConnectivityErrorCode.UNKNOWN
                    }
                is JsonDataException,
                is JsonEncodingException -> ConnectivityErrorCode.BACKEND_RESPONSE_INVALID
                is java.io.IOException -> ConnectivityErrorCode.NETWORK_UNREACHABLE
                else -> ConnectivityErrorCode.UNKNOWN
            }
            // For unclassified failures, include the exception class name so
            // the surfaced message is diagnosable instead of just "unknown".
            val detail = if (code == ConnectivityErrorCode.UNKNOWN) {
                "${throwable.javaClass.simpleName}: ${throwable.message ?: "no message"}"
            } else {
                throwable.message
            }
            return ConnectivityError(
                code = code,
                userMessage = code.userMessage,
                detail = detail
            )
        }
    }
}
