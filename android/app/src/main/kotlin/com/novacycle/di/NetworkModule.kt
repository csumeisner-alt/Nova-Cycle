package com.novacycle.di

import com.novacycle.BuildConfig
import com.novacycle.data.remote.NovaCycleApiService
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

/**
 * Hilt module providing all network-layer dependencies as singletons.
 * The API base URL comes from BuildConfig so it can be overridden per build variant
 * or set by the user (settings → stored in DataStore → injected via a qualifier).
 */
@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    /**
     * OkHttp client with:
     * - Logging interceptor (full body in DEBUG, none in RELEASE)
     * - 30s connect / read / write timeouts suitable for AI inference calls
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
     * This handles nullable fields, default values, and @Json name mapping.
     */
    @Provides
    @Singleton
    fun provideMoshi(): Moshi = Moshi.Builder()
        .addLast(KotlinJsonAdapterFactory())
        .build()

    /**
     * Retrofit configured with:
     * - Base URL from BuildConfig (default: http://10.0.2.2:8080/api/ for emulator)
     * - Moshi converter for JSON ↔ Kotlin data class mapping
     */
    @Provides
    @Singleton
    fun provideRetrofit(okHttpClient: OkHttpClient, moshi: Moshi): Retrofit =
        Retrofit.Builder()
            .baseUrl(BuildConfig.API_BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(MoshiConverterFactory.create(moshi))
            .build()

    @Provides
    @Singleton
    fun provideNovaCycleApiService(retrofit: Retrofit): NovaCycleApiService =
        retrofit.create(NovaCycleApiService::class.java)
}
