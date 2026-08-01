package com.novacycle.data.remote

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Observes Android network connectivity changes and emits `true` when a validated
 * internet connection becomes available, `false` when it is lost.
 *
 * Uses [ConnectivityManager.NetworkCallback] so ViewModels can react to
 * reconnection events without polling.  The flow starts with the current
 * connectivity state so collectors always have an initial value.
 */
@Singleton
open class NetworkMonitor @Inject constructor(
    @ApplicationContext private val context: Context
) {
    /**
     * Cold Flow of internet-reachability state.
     *
     * - Emits the current connected state immediately on collection.
     * - Emits `true` when any validated internet-capable network becomes available.
     * - Emits `false` when all such networks are lost.
     * - [distinctUntilChanged] ensures a cellular → Wi-Fi handoff (two transient
     *   [true] emissions) doesn't fire duplicate reconnect reloads.
     */
    open val isConnected: Flow<Boolean> = callbackFlow {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

        // Emit current state so the collector has an initial value right away.
        trySend(cm.isCurrentlyConnected())

        val callback = object : ConnectivityManager.NetworkCallback() {
            override fun onCapabilitiesChanged(
                network: Network,
                capabilities: NetworkCapabilities
            ) {
                trySend(capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET))
            }

            override fun onLost(network: Network) {
                // Re-check the overall device state — another network may still
                // be active (e.g. when Wi-Fi drops but cellular is available).
                trySend(cm.isCurrentlyConnected())
            }
        }

        val request = NetworkRequest.Builder()
            .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            .build()

        cm.registerNetworkCallback(request, callback)

        awaitClose { cm.unregisterNetworkCallback(callback) }
    }.distinctUntilChanged()

    private fun ConnectivityManager.isCurrentlyConnected(): Boolean {
        val caps = getNetworkCapabilities(activeNetwork) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }
}
