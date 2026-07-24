package com.novacycle.di

import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.stringPreferencesKey
import com.novacycle.BuildConfig
import com.novacycle.data.remote.NovaCycleApiService
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.firstOrNull
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.runBlocking
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

/**
 * Hilt module providing all network-layer dependencies as singletons.
 *
 * The API base URL is resolved at first injection time:
 *   1. Read the value stored in DataStore by the Settings screen (KEY_API_BASE_URL).
 *   2. Fall back to BuildConfig.API_BASE_URL if nothing has been saved yet.
 *
 * Changing the URL in Settings → tapping "Save API URL" persists the new value
 * to DataStore. The new URL is picked up automatically on the next app launch
 * without requiring an APK rebuild.
 */
@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    private val KEY_API_BASE_URL = stringPreferencesKey("api_base_url")

    /**
     * OkHttp client with:
     * - Logging interceptor (full body in DEBUG, none in RELEASE)
     * - 30 s connect / read / write timeouts suitable for AI inference calls
     */
    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient {
        val loggingLevel = if (BuildConfig.DEBUG) {
            HttpLoggingInterceptor.Level.BODY
        } else {
            HttpLoggingInterceptor.Level.NONE
        }
        val loggingInterceptor = HttpLoggingInterceptor().apply { level = loggingLevel }

        return OkHttpClient.Builder()
            .addInterceptor(loggingInterceptor)
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build()
    }

    /**
     * Moshi with KotlinJsonAdapterFactory for data class reflection support.
     * Handles nullable fields, default values, and @Json name mapping.
     */
    @Provides
    @Singleton
    fun provideMoshi(): Moshi = Moshi.Builder()
        .addLast(KotlinJsonAdapterFactory())
        .build()

    /**
     * Resolves the API base URL at singleton-creation time:
     *   - Prefers the user-saved URL from DataStore (set via Settings screen).
     *   - Falls back to BuildConfig.API_BASE_URL when no override exists.
     *
     * A blank string saved in DataStore is treated as "no override" so the
     * BuildConfig default is used instead.
     */
    @Provides
    @Singleton
    fun provideApiBaseUrl(dataStore: DataStore<Preferences>): String {
        val stored = runBlocking {
            dataStore.data
                .catch { emit(androidx.datastore.preferences.core.emptyPreferences()) }
                .map { prefs -> prefs[KEY_API_BASE_URL]?.takeIf { it.isNotBlank() } }
                .firstOrNull()
        }
        return stored ?: BuildConfig.API_BASE_URL
    }

    /**
     * Retrofit configured with:
     * - Base URL from DataStore (user-saved) or BuildConfig (default)
     * - Moshi converter for JSON ↔ Kotlin data class mapping
     */
    @Provides
    @Singleton
    fun provideRetrofit(
        okHttpClient: OkHttpClient,
        moshi: Moshi,
        apiBaseUrl: String
    ): Retrofit =
        Retrofit.Builder()
            .baseUrl(apiBaseUrl)
            .client(okHttpClient)
            .addConverterFactory(MoshiConverterFactory.create(moshi))
            .build()

    @Provides
    @Singleton
    fun provideNovaCycleApiService(retrofit: Retrofit): NovaCycleApiService =
        retrofit.create(NovaCycleApiService::class.java)
}
