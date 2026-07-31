package com.novacycle.data.remote

import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Transport-security rule: https:// anywhere; http:// only to private/LAN
 * destinations. Since the manifest network-security-config permits cleartext
 * globally, this validator is the only enforcement point — these tests are
 * the regression guard for "never plain HTTP to a public host".
 */
class ApiUrlValidatorTest {

    // ── persisted URL migration ────────────────────────────────────────────

    @Test
    fun `obsolete Replit development URL migrates to production`() {
        val production = "https://nova-cycle.replit.app/api/"
        assertTrue(
            ApiUrlResolver.resolve(
                "https://85621466-d083-4137-8a68-8de9779ab36a-00-lvz8z9d2rcc1.riker.replit.dev/api/",
                production
            ) == production
        )
    }

    @Test
    fun `custom server URL is preserved during migration`() {
        val custom = "https://api.example.com/api/"
        assertTrue(ApiUrlResolver.resolve(custom, "https://nova-cycle.replit.app/api/") == custom)
    }

    @Test
    fun `missing URL uses production default`() {
        val production = "https://nova-cycle.replit.app/api/"
        assertTrue(ApiUrlResolver.resolve(null, production) == production)
        assertTrue(ApiUrlResolver.resolve("   ", production) == production)
    }

    // ── https accepted anywhere ───────────────────────────────────────────

    @Test
    fun `https public host is accepted`() {
        assertNull(ApiUrlValidator.validate("https://nova-cycle.replit.app/api/"))
    }

    @Test
    fun `https LAN host is accepted`() {
        assertNull(ApiUrlValidator.validate("https://192.168.1.25:8000/api/"))
    }

    // ── http accepted only for private and local destinations ─────────────

    @Test
    fun `http localhost and loopback are accepted`() {
        assertNull(ApiUrlValidator.validate("http://localhost:8000/api/"))
        assertNull(ApiUrlValidator.validate("http://127.0.0.1:8000/"))
    }

    @Test
    fun `http RFC1918 private ranges are accepted`() {
        assertNull(ApiUrlValidator.validate("http://192.168.1.25:8000"))
        assertNull(ApiUrlValidator.validate("http://10.0.0.7/api/"))
        assertNull(ApiUrlValidator.validate("http://172.16.0.1:8080"))
        assertNull(ApiUrlValidator.validate("http://172.31.255.254"))
    }

    @Test
    fun `http link-local, CGNAT, mDNS and single-label hosts are accepted`() {
        assertNull(ApiUrlValidator.validate("http://169.254.10.10"))
        assertNull(ApiUrlValidator.validate("http://100.64.3.2:8000"))
        assertNull(ApiUrlValidator.validate("http://mymac.local:8000"))
        assertNull(ApiUrlValidator.validate("http://devbox:8000"))
    }

    @Test
    fun `http public hosts are rejected with a helpful message`() {
        val error = ApiUrlValidator.validate("http://nova-cycle.replit.app/api/")
        assertNotNull(error)
        assertTrue(error!!.contains("https://"))
        assertNotNull(ApiUrlValidator.validate("http://example.com"))
        assertNotNull(ApiUrlValidator.validate("http://8.8.8.8"))
        // Public-adjacent ranges just OUTSIDE the private blocks
        assertNotNull(ApiUrlValidator.validate("http://172.15.0.1"))
        assertNotNull(ApiUrlValidator.validate("http://172.32.0.1"))
        assertNotNull(ApiUrlValidator.validate("http://100.128.0.1"))
        assertNotNull(ApiUrlValidator.validate("http://11.0.0.1"))
        assertNotNull(ApiUrlValidator.validate("http://193.168.1.1"))
    }

    // ── basic validation still works ───────────────────────────────────────

    @Test
    fun `empty, schemeless and malformed URLs are rejected`() {
        assertNotNull(ApiUrlValidator.validate(""))
        assertNotNull(ApiUrlValidator.validate("   "))
        assertNotNull(ApiUrlValidator.validate("nova-cycle.replit.app/api"))
        assertNotNull(ApiUrlValidator.validate("ftp://192.168.1.1"))
        assertNotNull(ApiUrlValidator.validate("http://"))
    }

    // ── host classifier edge cases ─────────────────────────────────────────

    @Test
    fun `ipv6 loopback, link-local and ULA are private`() {
        assertTrue(ApiUrlValidator.isPrivateOrLocalHost("::1"))
        assertTrue(ApiUrlValidator.isPrivateOrLocalHost("[::1]"))
        assertTrue(ApiUrlValidator.isPrivateOrLocalHost("fe80::1"))
        assertTrue(ApiUrlValidator.isPrivateOrLocalHost("fd12:3456::1"))
        assertFalse(ApiUrlValidator.isPrivateOrLocalHost("2001:4860:4860::8888"))
    }

    // ── bypass vectors ─────────────────────────────────────────────────────

    @Test
    fun `userinfo trick cannot smuggle a public host past the private check`() {
        // java.net.URL.getHost() strips userinfo, so the real host is evil.com
        assertNotNull(ApiUrlValidator.validate("http://192.168.1.1@evil.com"))
    }

    @Test
    fun `numeric and hex IP encodings are rejected over http`() {
        assertNotNull(ApiUrlValidator.validate("http://2130706433"))      // 127.0.0.1 as int
        assertNotNull(ApiUrlValidator.validate("http://134744072"))       // 8.8.8.8 as int
        assertNotNull(ApiUrlValidator.validate("http://0x7f000001"))      // hex form
        assertNotNull(ApiUrlValidator.validate("http://0x7f.0.0.1"))      // hex octet
        assertNotNull(ApiUrlValidator.validate("http://0177.0.0.1"))      // octal-looking (parses as 177 → not 127/10/etc.)
    }

    @Test
    fun `case and trailing-dot variants stay on the safe side`() {
        assertNull(ApiUrlValidator.validate("http://LOCALHOST:8000"))     // case-insensitive
        assertNull(ApiUrlValidator.validate("http://DevBox:8000"))
        assertNotNull(ApiUrlValidator.validate("http://localhost.:8000")) // trailing dot → treated as public (safe)
        assertNotNull(ApiUrlValidator.validate("HTTP://example.com"))     // scheme case → fails scheme prefix check (safe)
    }

    @Test
    fun `ipv4-mapped ipv6 of a public address is not private`() {
        assertFalse(ApiUrlValidator.isPrivateOrLocalHost("::ffff:8.8.8.8"))
        assertFalse(ApiUrlValidator.isPrivateOrLocalHost("[::ffff:8.8.8.8]"))
    }

    @Test
    fun `almost-IP strings are treated as public names, not IPs`() {
        // Not a valid IPv4 → falls through to the DNS-name rule (public)
        assertFalse(ApiUrlValidator.isPrivateOrLocalHost("192.168.1"))
        assertFalse(ApiUrlValidator.isPrivateOrLocalHost("192.168.1.999"))
        assertFalse(ApiUrlValidator.isPrivateOrLocalHost("10.evil.com"))
    }
}
