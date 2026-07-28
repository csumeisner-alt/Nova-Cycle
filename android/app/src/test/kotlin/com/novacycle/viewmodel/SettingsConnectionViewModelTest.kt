package com.novacycle.viewmodel

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.emptyPreferences
import com.novacycle.data.remote.ConnectivityErrorMapper
import com.novacycle.data.remote.NetworkStatusChecker
import com.novacycle.data.repository.NovaCycleRepository
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test


/**
 * ViewModel-level regression tests for the Settings connection test.
 *
 * The original bug: testConnection() executed a synchronous OkHttp call on the
 * main dispatcher, so on-device it always threw NetworkOnMainThreadException
 * and reported "Connection failed for an unknown reason". These tests drive
 * testConnection() end-to-end against a real local HTTP server, with the
 * injected [SettingsViewModel.ioDispatcher] seam exercised for real I/O.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class SettingsConnectionViewModelTest {

    private val mainDispatcher = StandardTestDispatcher()

    /**
     * Minimal blocking HTTP server on a ServerSocket: /api/healthz → 200,
     * anything else → 500. com.sun.net.httpserver is not on the Android
     * unit-test classpath, so we hand-roll the two responses we need.
     */
    private lateinit var serverSocket: java.net.ServerSocket
    @Volatile private var running = true
    private var serverThread: Thread? = null
    private val port: Int get() = serverSocket.localPort

    @Before
    fun setUp() {
        // The failure path logs via android.util.Log, which is not available
        // in local unit tests — stub it out.
        io.mockk.mockkStatic(android.util.Log::class)
        every { android.util.Log.w(any<String>(), any<String>()) } returns 0
        Dispatchers.setMain(mainDispatcher)
        serverSocket = java.net.ServerSocket(0, 50, java.net.InetAddress.getByName("127.0.0.1"))
        running = true
        serverThread = Thread {
            while (running) {
                try {
                    val socket = serverSocket.accept()
                    socket.use { s ->
                        val reader = s.getInputStream().bufferedReader()
                        val requestLine = reader.readLine() ?: return@use
                        // Drain headers until the blank line.
                        while (true) {
                            val line = reader.readLine() ?: break
                            if (line.isEmpty()) break
                        }
                        val ok = requestLine.contains("/api/healthz")
                        val body = """{"status":"ok"}"""
                        val response = if (ok) {
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n" +
                                "Content-Length: ${body.length}\r\nConnection: close\r\n\r\n$body"
                        } else {
                            "HTTP/1.1 500 Internal Server Error\r\n" +
                                "Content-Length: 0\r\nConnection: close\r\n\r\n"
                        }
                        s.getOutputStream().write(response.toByteArray())
                        s.getOutputStream().flush()
                    }
                } catch (_: Exception) {
                    // Socket closed during teardown — exit loop.
                    if (!running) return@Thread
                }
            }
        }.apply { isDaemon = true; start() }
    }

    @After
    fun tearDown() {
        running = false
        runCatching { serverSocket.close() }
        serverThread?.join(2000)
        Dispatchers.resetMain()
    }

    private fun buildViewModel(): SettingsViewModel {
        val dataStore = mockk<DataStore<Preferences>>()
        every { dataStore.data } returns flowOf(emptyPreferences())
        val networkChecker = mockk<NetworkStatusChecker>()
        every { networkChecker.isOffline() } returns false
        return SettingsViewModel(
            dataStore = dataStore,
            repository = mockk<NovaCycleRepository>(relaxed = true),
            connectivityErrorMapper = ConnectivityErrorMapper(networkChecker),
            appContext = mockk(relaxed = true)
        ).also {
            // Real IO dispatcher: proves the network call works off the main thread.
            it.ioDispatcher = Dispatchers.IO
        }
    }

    private fun awaitTerminalState(vm: SettingsViewModel): ConnectionTestState {
        // The IO leg runs on a real dispatcher, but its continuation resumes
        // on the test main dispatcher — pump the scheduler while waiting in
        // real time for the network round-trip to finish.
        repeat(400) {
            mainDispatcher.scheduler.runCurrent()
            val s = vm.connectionTestState.value
            if (s is ConnectionTestState.Success || s is ConnectionTestState.Failure) return s
            Thread.sleep(25)
        }
        return vm.connectionTestState.value
    }

    @Test
    fun `healthy backend reports Connected`() = runTest(mainDispatcher) {
        val vm = buildViewModel()
        vm.testConnection("http://127.0.0.1:$port/api/")
        val state = awaitTerminalState(vm)
        assertTrue("expected Success, got $state", state is ConnectionTestState.Success)
        assertTrue((state as ConnectionTestState.Success).message.contains("Connected"))
    }

    @Test
    fun `server error reports classified HTTP failure not unknown`() = runTest(mainDispatcher) {
        val vm = buildViewModel()
        vm.testConnection("http://127.0.0.1:$port/broken/")
        val state = awaitTerminalState(vm)
        assertTrue("expected Failure, got $state", state is ConnectionTestState.Failure)
        assertTrue((state as ConnectionTestState.Failure).message.contains("HTTP 500"))
    }

    @Test
    fun `unreachable server reports classified failure with detail`() = runTest(mainDispatcher) {
        val vm = buildViewModel()
        // Port 1 on localhost: connection refused immediately.
        vm.testConnection("http://127.0.0.1:1/api/")
        val state = awaitTerminalState(vm)
        assertTrue("expected Failure, got $state", state is ConnectionTestState.Failure)
        val msg = (state as ConnectionTestState.Failure).message
        assertTrue("message should be classified, was: $msg", !msg.contains("unknown reason"))
    }

    @Test
    fun `state passes through Testing before terminal state`() = runTest(mainDispatcher) {
        val vm = buildViewModel()
        assertEquals(ConnectionTestState.Idle, vm.connectionTestState.value)
        vm.testConnection("http://127.0.0.1:$port/api/")
        assertEquals(ConnectionTestState.Testing, vm.connectionTestState.value)
        val state = awaitTerminalState(vm)
        assertTrue(state is ConnectionTestState.Success)
    }
}
