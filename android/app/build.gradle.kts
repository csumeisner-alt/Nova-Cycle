import java.util.Base64

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.compose.compiler)
    alias(libs.plugins.hilt.android)
    // alias(libs.plugins.google.services)  // Uncomment when google-services.json is added
    id("kotlin-kapt")
    id("kotlin-parcelize")
}

android {
    namespace = "com.novacycle"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.novacycle"
        minSdk = 26
        targetSdk = 35
        versionCode = 4
        versionName = "1.3"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }

        // Backend API URL — points to the live Replit backend over HTTPS.
        // This is the single source of truth for the default API URL; all in-app
        // fallbacks reference BuildConfig.API_BASE_URL. Users can override it at
        // runtime from the Settings screen without rebuilding the APK.
        buildConfigField("String", "API_BASE_URL", "\"https://nova-cycle.replit.app/api/\"")
    }

    signingConfigs {
        create("release") {
            // Credentials are injected by the CI environment (GitHub Actions secrets).
            // Locally, set these four env vars or the build falls back to debug signing.
            val keystoreBase64 = System.getenv("KEYSTORE_BASE64")
            val keystorePassword = System.getenv("KEYSTORE_PASSWORD")
            val keyAlias = System.getenv("KEY_ALIAS")
            val keyPassword = System.getenv("KEY_PASSWORD")

            if (keystoreBase64 != null && keystorePassword != null &&
                keyAlias != null && keyPassword != null
            ) {
                // Decode the base64 keystore to a temp file at build time
                val keystoreFile = layout.buildDirectory.file("novacycle-release.p12").get().asFile
                keystoreFile.parentFile.mkdirs()
                keystoreFile.writeBytes(
                    Base64.getDecoder().decode(keystoreBase64)
                )
                storeFile = keystoreFile
                storePassword = keystorePassword
                this.keyAlias = keyAlias
                this.keyPassword = keyPassword
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            // CI injects the protected signing key. Never substitute the
            // debug key here: Android rejects updates when the signing key
            // changes, even when the package name is identical.
            signingConfig = signingConfigs.getByName("release")
        }
        debug {
            // Debug keeps API URL readable, minification off.
            // Android Gradle plugin auto-generates ~/.android/debug.keystore on first build —
            // no manual keystore setup required for debug APKs.
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    // Unit tests (data-freshness contract behind the "last updated X ago" label,
    // HealthViewModel banner state machine)
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.1")
    testImplementation("io.mockk:mockk:1.13.11")

    // Material Components (provides Theme.Material3 used in AndroidManifest/themes)
    implementation(libs.material)

    // Compose Material Icons Extended — full icon set including Speed, ShowChart, Timeline, BarChart
    implementation(libs.material.icons.extended)

    // Compose BOM — manages all compose library versions in sync
    val composeBom = platform(libs.compose.bom)
    implementation(composeBom)
    debugImplementation(composeBom)

    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    // Compose Material (M2) — pullRefresh modifier + PullRefreshIndicator (no M3 equivalent in this BOM)
    implementation(libs.compose.material)
    implementation(libs.compose.foundation)
    implementation(libs.compose.animation)
    debugImplementation(libs.compose.ui.tooling)

    // Activity + Lifecycle
    implementation(libs.activity.compose)
    implementation(libs.core.splashscreen)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)

    // Hilt DI
    implementation(libs.hilt.android)
    kapt(libs.hilt.compiler)
    implementation(libs.hilt.navigation.compose)

    // Room local database
    implementation(libs.room.runtime)
    implementation(libs.room.ktx)
    kapt(libs.room.compiler)

    // Retrofit + OkHttp networking
    implementation(libs.retrofit)
    implementation(libs.retrofit.converter.moshi)
    implementation(libs.okhttp)
    implementation(libs.okhttp.logging.interceptor)

    // Moshi JSON parsing
    implementation(libs.moshi)
    implementation(libs.moshi.kotlin)

    // Firebase Messaging. The SDK is safe to ship before project configuration:
    // without google-services.json Firebase initialization is caught and the
    // app continues without push. Add the file and enable the Google Services
    // plugin before producing a notification-enabled release.
    implementation(platform(libs.firebase.bom))
    implementation(libs.firebase.messaging)

    // DataStore for settings persistence
    implementation(libs.datastore.preferences)

    // Coroutines
    implementation(libs.coroutines.android)

    // Navigation
    implementation(libs.navigation.compose)
}

// Allow references to generated code
kapt {
    correctErrorTypes = true
}
