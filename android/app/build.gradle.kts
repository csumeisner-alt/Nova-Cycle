plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
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
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }

        // Backend API URL — points to the live Replit backend.
        // For local development against an emulator, change to "http://10.0.2.2:8080/api/"
        // For a real device against the Replit backend, keep the https:// URL below.
        buildConfigField("String", "API_BASE_URL", "\"https://85621466-d083-4137-8a68-8de9779ab36a-00-lvz8z9d2rcc1.riker.replit.dev/api/\"")
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
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

    composeOptions {
        // Kotlin 2.0.0 uses compose compiler plugin 1.5.14
        kotlinCompilerExtensionVersion = "1.5.14"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    // Compose BOM — manages all compose library versions in sync
    val composeBom = platform(libs.compose.bom)
    implementation(composeBom)
    debugImplementation(composeBom)

    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.foundation)
    implementation(libs.compose.animation)
    debugImplementation(libs.compose.ui.tooling)

    // Activity + Lifecycle
    implementation(libs.activity.compose)
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

    // Firebase — uncomment these two lines once google-services.json is in android/app/
    // and re-enable alias(libs.plugins.google.services) in the plugins block above.
    // implementation(platform(libs.firebase.bom))
    // implementation(libs.firebase.messaging)

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
