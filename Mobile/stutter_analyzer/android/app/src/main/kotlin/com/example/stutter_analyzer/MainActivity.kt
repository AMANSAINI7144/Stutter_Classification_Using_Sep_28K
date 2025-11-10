package com.example.stutter_analyzer

import android.content.Context
import android.os.Bundle
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import kotlin.random.Random

class MainActivity : FlutterActivity() {
    private val CHANNEL = "stutter/model"
    private var module: Module? = null
    private val exec = Executors.newSingleThreadExecutor()

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // load model on background thread
        exec.execute {
            try {
                initModel(this)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "testModel" -> {
                    exec.execute {
                        try {
                            val out = runTestModel()
                            val map = HashMap<String, Any>()
                            map["ok"] = true
                            map["output"] = out.toList()
                            runOnUiThread { result.success(map) }
                        } catch (e: Exception) {
                            e.printStackTrace()
                            runOnUiThread { result.error("err", e.message, null) }
                        }
                    }
                }
                "runModelFromPath" -> {
                    val wavPath = call.argument<String>("wav_path")
                    if (wavPath == null) {
                        result.error("no_path", "wav_path argument missing", null)
                        return@setMethodCallHandler
                    }
                    exec.execute {
                        try {
                            // Not implemented: put your WAV decode & inference here later
                            runOnUiThread { result.error("not_impl", "runModelFromPath not implemented yet", null) }
                        } catch (e: Exception) {
                            e.printStackTrace()
                            runOnUiThread { result.error("err", e.message, null) }
                        }
                    }
                }
                else -> result.notImplemented()
            }
        }
    }

    private fun initModel(context: Context) {
        if (module != null) return
        val assetName = "model_torchscript.pt"
        val outFile = File(context.filesDir, assetName)
        if (!outFile.exists()) {
            context.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        module = Module.load(outFile.absolutePath)
    }

    private fun runTestModel(): FloatArray {
        val mod = module ?: throw RuntimeException("Model not loaded")
        val L = 16000
        val arr = FloatArray(L) { Random.nextFloat() * 2f - 1f }
        val input = Tensor.fromBlob(arr, longArrayOf(1, L.toLong()))
        val outIVal = mod.forward(IValue.from(input))
        val outTensor = if (outIVal.isTuple) outIVal.toTuple()[0].toTensor() else outIVal.toTensor()
        return outTensor.dataAsFloatArray
    }

    override fun onDestroy() {
        super.onDestroy()
        exec.shutdownNow()
    }
}
