// lib/services/native_inference.dart
import 'dart:async';
import 'package:flutter/services.dart';

class NativeInference {
  static const MethodChannel _ch = MethodChannel('stutter/model');

  /// Run model by providing a file path to a WAV.
  /// Returns a Map with model outputs (deserialize-friendly).
  static Future<Map<String, dynamic>> runModelFromPath(String wavPath, {Duration timeout = const Duration(seconds: 60)}) async {
    try {
      final res = await _ch.invokeMapMethod<String, dynamic>('runModelFromPath', {'wav_path': wavPath}).timeout(timeout);
      if (res == null) throw Exception('Null response from native model');
      // ensure Map<String, dynamic>
      return Map<String, dynamic>.from(res);
    } on MissingPluginException {
      throw Exception('Native inference plugin not implemented. Implement the Kotlin MethodChannel "stutter/model" with method "runModelFromPath".');
    } on TimeoutException {
      throw Exception('Model inference timed out.');
    } catch (e) {
      rethrow;
    }
  }

  /// Smoke test: call native testModel method which runs model on synthetic input
  static Future<Map<String, dynamic>> testModel({Duration timeout = const Duration(seconds: 20)}) async {
    try {
      final res = await _ch.invokeMapMethod<String, dynamic>('testModel').timeout(timeout);
      if (res == null) throw Exception('Null response from native testModel');
      return Map<String, dynamic>.from(res);
    } on MissingPluginException {
      throw Exception('Native testModel not implemented. Implement the Kotlin MethodChannel "stutter/model" with method "testModel".');
    } on TimeoutException {
      throw Exception('Native testModel timed out.');
    } catch (e) {
      rethrow;
    }
  }
}
