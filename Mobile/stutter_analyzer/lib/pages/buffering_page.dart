// lib/pages/buffering_page.dart
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'result_page.dart';

class BufferingPage extends StatefulWidget {
  final String wavPath;
  const BufferingPage({super.key, required this.wavPath});

  @override
  State<BufferingPage> createState() => _BufferingPageState();
}

class _BufferingPageState extends State<BufferingPage> {
  static const MethodChannel _ch = MethodChannel('stutter/model');

  @override
  void initState() {
    super.initState();
    // Kick off inference after the first frame so UI shows spinner.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _runInference();
    });
  }

  Future<void> _runInference() async {
    try {
      // First try the main method that will accept a wav path (native Kotlin)
      final dynamic res = await _ch.invokeMethod('runModelFromPath', {'wav_path': widget.wavPath});
      _handleResult(res);
    } on PlatformException catch (e) {
      // If native method not implemented yet, fallback to testModel
      if (e.code == 'not_impl' || (e.message?.contains('not_impl') ?? false)) {
        try {
          final dynamic testRes = await _ch.invokeMethod('testModel');
          _handleResult(testRes);
        } catch (e2) {
          _showError('Native testModel failed: $e2');
        }
      } else {
        _showError('Platform error: ${e.message}');
      }
    } catch (e) {
      _showError('Unexpected error: $e');
    }
  }

  void _handleResult(dynamic res) {
    // The Kotlin side returns either:
    //  - a Map: {'ok': true, 'output': [list of numbers]}
    //  - or a flattened array/list or string (fallback). Handle common cases.
    List<double> probs = [];

    if (res == null) {
      _showError('Model returned no result.');
      return;
    }

    if (res is Map) {
      final dynamic out = res['output'];
      if (out is List) {
        probs = out.map((e) => (e is num) ? e.toDouble() : double.tryParse('$e') ?? 0.0).toList();
      }
    } else if (res is List) {
      probs = res.map((e) => (e is num) ? e.toDouble() : double.tryParse('$e') ?? 0.0).toList();
    } else if (res is String) {
      // try to parse comma separated floats
      probs = res.split(',').map((s) => double.tryParse(s.trim()) ?? 0.0).toList();
    }

    // If no probs extracted, show error
    if (probs.isEmpty) {
      _showError('Could not parse model output.');
      return;
    }

    // Navigate to result page
    if (!mounted) return;
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => ResultPage(probabilities: probs)),
    );
  }

  void _showError(String msg) {
    if (!mounted) return;
    // show a dialog and pop back after user taps OK
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Inference error'),
        content: Text(msg),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.pop(context); // dismiss dialog
              Navigator.pop(context); // go back to choose page
            },
            child: const Text('OK'),
          )
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Analyzing...'),
        centerTitle: true,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: const [
            CircularProgressIndicator(),
            SizedBox(height: 18),
            Text('Running model — please wait'),
          ],
        ),
      ),
    );
  }
}
