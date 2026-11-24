plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.chisnova.mercury"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.chisnova.mercury"
        minSdk = 24
        targetSdk = 34
        buildConfigField("String", "FULL_UNLOCK_PRODUCT_ID", "\"com.chisnova.mercury.full_unlock\"")
        // Play Console > Monetization setup > licensing public key (not a secret).
        val billingKey = providers.gradleProperty("PLAY_BILLING_PUBLIC_KEY").orElse("").get()
        require(billingKey.matches(Regex("[A-Za-z0-9+/=]*"))) { "PLAY_BILLING_PUBLIC_KEY must be Base64" }
        buildConfigField("String", "PLAY_BILLING_PUBLIC_KEY", "\"$billingKey\"")
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    // Ensure native libraries in src/main/libs are packaged into the APK
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/libs")
            // Shared with iOS; edit legal/policies.json once for both apps.
            assets.srcDir("../../legal")
        }
    }
}

dependencies {
    implementation("com.android.billingclient:billing:8.3.0")

    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.constraintlayout:constraintlayout:2.2.1")
    implementation("com.google.android.material:material:1.12.0")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
    androidTestImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:core:1.6.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
    implementation("com.google.code.gson:gson:2.10.1")

    // ExoPlayer for opus playback from assets
    implementation("com.google.android.exoplayer:exoplayer-core:2.19.1")
    implementation("com.google.android.exoplayer:exoplayer-ui:2.19.1")

    // Coroutines — for offloading I/O off the main thread
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")

    // RecyclerView — sentence list virtualization
    implementation("androidx.recyclerview:recyclerview:1.3.2")
}
