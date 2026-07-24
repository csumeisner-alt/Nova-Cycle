# NovaCycle ProGuard rules

# ── General attributes needed by reflection-based libraries ──
-keepattributes *Annotation*
-keepattributes Signature, InnerClasses, EnclosingMethod
-keepattributes RuntimeVisibleAnnotations, RuntimeVisibleParameterAnnotations
-keepattributes SourceFile, LineNumberTable
-renamesourcefileattribute SourceFile

# ── Keep Kotlin metadata and coroutines ──
-keep class kotlin.Metadata { *; }
-keep class kotlin.coroutines.** { *; }
-keep class kotlinx.coroutines.** { *; }
-dontwarn kotlinx.coroutines.**
-keepclassmembernames class kotlinx.** {
    volatile <fields>;
}

# ── AndroidX / Compose / Navigation ──
-keep class androidx.** { *; }
-dontwarn androidx.**
-keep class androidx.compose.** { *; }
-keep class androidx.navigation.** { *; }
-keep class androidx.hilt.navigation.compose.** { *; }
-keep class androidx.lifecycle.** { *; }
-keep class androidx.activity.** { *; }
-keep class androidx.datastore.** { *; }
-keep class androidx.datastore.preferences.** { *; }
-keep class androidx.room.** { *; }

# ── Hilt / Dagger generated components ──
-keep class dagger.hilt.** { *; }
-keep class javax.inject.** { *; }
-keep class * extends dagger.hilt.internal.GeneratedComponentManagerHolder { *; }
-keep class * extends dagger.hilt.android.HiltActivity { *; }
-keep class * extends dagger.hilt.android.HiltAndroidApp { *; }
-keep class * extends android.app.Application { *; }
-keepclassmembers class * {
    @dagger.hilt.android.AndroidEntryPoint <init>(...);
}
-keepclassmembers @dagger.hilt.android.AndroidEntryPoint class * {
    @javax.inject.Inject <fields>;
}
-keep class dagger.hilt.android.internal.managers.** { *; }

# ── NovaCycle app classes ──
-keep class com.novacycle.** { *; }
-keepclassmembers class com.novacycle.** { *; }
-dontwarn com.novacycle.**

# ── Keep enum values() / valueOf() safe ──
-keepclassmembers enum com.novacycle.** {
    public static **[] values();
    public static ** valueOf(java.lang.String);
    <fields>;
}
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# ── Retrofit ──
-keep class retrofit2.** { *; }
-keepclassmembers,allowshrinking,allowobfuscation interface * {
    @retrofit2.http.* <methods>;
}
-dontwarn org.codehaus.mojo.animal_sniffer.IgnoreJRERequirement
-dontwarn javax.annotation.**
-dontwarn kotlin.Unit
-dontwarn retrofit2.KotlinExtensions
-dontwarn retrofit2.KotlinExtensions$*

# ── Moshi ──
-keep class com.squareup.moshi.** { *; }
-keep @com.squareup.moshi.JsonQualifier interface *
-keepclasseswithmembers class * {
    @com.squareup.moshi.* <methods>;
}
-keep class kotlin.reflect.jvm.internal.** { *; }
-dontwarn com.squareup.moshi.**

# ── Room ──
-keep class * extends androidx.room.RoomDatabase
-keep @androidx.room.Entity class *
-keep @androidx.room.Dao class *
-keep class * extends androidx.room.RoomDatabase { *; }
-dontwarn androidx.room.paging.**

# ── OkHttp / Okio ──
-keep class okhttp3.** { *; }
-keep class okio.** { *; }
-dontwarn okhttp3.**
-dontwarn okio.**
-dontwarn org.bouncycastle.**
-dontwarn org.conscrypt.**
-dontwarn org.openjsse.**

# ── Firebase (disabled, but keep rules safe) ──
-keep class com.google.firebase.** { *; }
-dontwarn com.google.firebase.**

# ── Material Icons Extended ──
-keep class androidx.compose.material.icons.** { *; }
-dontwarn androidx.compose.material.icons.**

# ── Logging / Crash safety ──
-dontwarn org.slf4j.**
-dontwarn android.util.**
