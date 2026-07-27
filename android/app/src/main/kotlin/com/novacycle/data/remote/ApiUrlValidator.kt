package com.novacycle.data.remote

import java.net.MalformedURLException
import java.net.URL

/**
 * Validates user-entered API base URLs with a transport-security rule:
 *
 *  - `https://` is accepted for any host (default and preferred).
 *  - `http://` (cleartext) is accepted ONLY for private/LAN destinations —
 *    loopback, RFC-1918 ranges, link-local, `.local` mDNS names, and bare
 *    single-label hostnames (e.g. `http://mylaptop:8000`) that can only
 *    resolve on the local network.
 *
 * Android's network security config cannot express "cleartext for private IP
 * ranges only", so the manifest config permits cleartext globally and THIS
 * validator is the enforcement point: a plain-HTTP URL to a public internet
 * host is rejected before it can ever be saved or tested.
 */
object ApiUrlValidator {

    /** @return an error message, or null when the URL is acceptable. */
    fun validate(url: String): String? {
        val trimmed = url.trim()
        if (trimmed.isEmpty()) return "URL must not be empty"
        if (!trimmed.startsWith("http://") && !trimmed.startsWith("https://")) {
            return "URL must start with http:// or https://"
        }
        val parsed = try {
            URL(trimmed)
        } catch (e: MalformedURLException) {
            return "Invalid URL: ${e.message}"
        }
        if (parsed.host.isNullOrBlank()) return "URL must include a host"

        if (parsed.protocol == "http" && !isPrivateOrLocalHost(parsed.host)) {
            return "Plain http:// is only allowed for local network addresses " +
                "(e.g. 192.168.x.x, 10.x.x.x, localhost). Use https:// for " +
                "internet servers."
        }
        return null
    }

    /**
     * True when [host] can only be a local/private destination:
     * loopback, RFC-1918 private IPv4 ranges, link-local, CGNAT (100.64/10),
     * IPv6 loopback/ULA/link-local, `.local` mDNS names, and single-label
     * hostnames (no dots — not resolvable on the public internet).
     */
    fun isPrivateOrLocalHost(rawHost: String): Boolean {
        val host = rawHost.trim().trimStart('[').trimEnd(']').lowercase()
        if (host.isEmpty()) return false

        // Names
        if (host == "localhost") return true
        if (host.endsWith(".local")) return true          // mDNS
        // Single-label LAN hostname (e.g. "devbox"). Must look like a DNS
        // label starting with a letter — this rejects alternate numeric IP
        // encodings such as "2130706433" (= 127.0.0.1 as a 32-bit integer,
        // which could equally encode a PUBLIC address) and hex forms "0x…".
        if (!host.contains('.') && !host.contains(':')) {
            return SINGLE_LABEL_HOSTNAME.matches(host) && !host.startsWith("0x")
        }

        // IPv6
        if (host.contains(':')) {
            return host == "::1" ||
                host.startsWith("fe80:") ||               // link-local
                host.startsWith("fc") || host.startsWith("fd")  // unique-local fc00::/7
        }

        // IPv4 — only treat as an IP when all four octets parse
        val octets = host.split('.')
        if (octets.size == 4 && octets.all { (it.toIntOrNull() ?: -1) in 0..255 }) {
            val a = octets[0].toInt()
            val b = octets[1].toInt()
            return a == 127 ||                                   // loopback
                a == 10 ||                                       // 10/8
                (a == 192 && b == 168) ||                        // 192.168/16
                (a == 172 && b in 16..31) ||                     // 172.16/12
                (a == 169 && b == 254) ||                        // link-local
                (a == 100 && b in 64..127)                       // CGNAT 100.64/10
        }

        // Multi-label DNS name (e.g. example.com) — treat as public.
        // This also covers alternate IPv4 textual encodings (octal/hex octets
        // like "0x7f.0.0.1", out-of-range octets, dotted partial forms):
        // anything that doesn't parse as a plain dotted-quad falls through to
        // "public" and therefore requires https.
        return false
    }

    /** DNS-like LAN label: starts with a letter, then letters/digits/hyphens. */
    private val SINGLE_LABEL_HOSTNAME = Regex("^[a-z]([a-z0-9-]*[a-z0-9])?$")
}
