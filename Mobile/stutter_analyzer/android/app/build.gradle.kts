plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.stutter_analyzer"
    compileSdk = flutter.compileSdkVersion

    // ✅ Fix NDK version mismatch (use version 27 as required by plugins)
    ndkVersion = "27.0.12077973"

    compileOptions {
        // ✅ Ensure Java 11 compatibility
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_11.toString()
    }

    defaultConfig {
        // ✅ Unique application ID
        applicationId = "com.example.stutter_analyzer"

        // ✅ Flutter’s configured values
        minSdk = 21
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // ✅ Keep debug signing for now (can be updated later for production)
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

dependencies {
    // ✅ Core PyTorch Mobile dependency (CPU-only)
    implementation("org.pytorch:pytorch_android:1.13.1")

    // ✅ TorchVision (optional – used for preprocessing/transforms)
    implementation("org.pytorch:pytorch_android_torchvision:1.13.1")

    // ✅ Add other future dependencies here if needed
}

flutter {
    source = "../.."
}
