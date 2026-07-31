package com.novacycle.data.remote

import java.net.MalformedURLException
import java.net.URL

/**
 * Resolves an API URL saved by an earlier app build.
 *
 * The first production APKs used a temporary Replit development hostname.
 * Android preserves DataStore through an APK update, so changing
 * BuildConfig.API_BASE_URL alone does not repair an installation that already
 * saved that hostname. Only NovaCycle's obsolete Replit development hosts are
 * migrated; user-configured private or third-party servers are left alone.
 */
object ApiUrlResolver {

    /**
     * Returns the stored URL when it is still intentional, or [defaultUrl]
     * when the stored URL is a known obsolete NovaCycle development address.
     */
    fun resolve(storedUrl: String?, defaultUrl: String): String {
        val stored = storedUrl?.trim()?.takeIf { it.isNotEmpty() } ?: return defaultUrl
        return if (isObsoleteNovaCycleUrl(stored)) defaultUrl else stored
    }

    /**
     * The old app used a temporary *.replit.dev hostname. These hosts are
     * workspace-scoped and can stop resolving; they must never remain the
     * production app's persisted backend address.
     */
    fun isObsoleteNovaCycleUrl(rawUrl: String): Boolean {
        val parsed = try {
            URL(rawUrl.trim())
        } catch (_: MalformedURLException) {
            return false
        }
        return parsed.protocol == "https" &&
            parsed.host.lowercase().endsWith(".replit.dev")
    }
}