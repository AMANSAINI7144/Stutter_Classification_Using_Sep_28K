// lib/pages/choose_file_page.dart
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'buffering_page.dart';

class ChooseFilePage extends StatefulWidget {
  static const routeName = '/';
  const ChooseFilePage({super.key});

  @override
  State<ChooseFilePage> createState() => _ChooseFilePageState();
}

class _ChooseFilePageState extends State<ChooseFilePage> {
  String? _pickedPath;
  Uint8List? _pickedBytes;

  Future<void> _pickWav() async {
    try {
      final res = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['wav'],
        allowMultiple: false,
        withData: true, // ensures bytes are available on web
      );
      if (res == null) return;

      // on web, path is always null; use bytes
      final pf = res.files.single;
      if (kIsWeb || pf.path == null) {
        // we have bytes available because withData:true
        if (pf.bytes == null) {
          // defensive - this should not happen with withData:true
          _showMessage('Could not read file bytes on web.');
          return;
        }
        setState(() {
          _pickedBytes = pf.bytes;
          _pickedPath = null;
        });

        // we cannot run native on web; show friendly message
        _showWebNotSupportedDialog();
        return;
      }

      // Non-web platforms (Android/iOS) will have a real path
      final path = pf.path!;
      setState(() {
        _pickedPath = path;
        _pickedBytes = null;
      });

      // Navigate to buffering page and pass the path
      if (!mounted) return;
      Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => BufferingPage(wavPath: path)),
      );
    } catch (e) {
      _showMessage('Error picking file: $e');
    }
  }

  void _showWebNotSupportedDialog() {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Web not supported'),
        content: const Text(
          'On the web the app cannot run native on-device inference. '
              'Please run this app on an Android device/emulator to run the model. '
              'You can still use the UI here, but file paths are not available on web.',
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK')),
        ],
      ),
    );
  }

  void _showMessage(String txt) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(txt)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Choose WAV'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 36),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(height: 24),
            const Text(
              'Select a .wav file to analyze (16kHz mono recommended).',
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 28),
            ElevatedButton.icon(
              onPressed: _pickWav,
              icon: const Icon(Icons.upload_file),
              label: const Text('Choose WAV file'),
            ),
            const SizedBox(height: 18),
            if (_pickedPath != null) ...[
              const Text('Selected file:'),
              const SizedBox(height: 6),
              Text(
                _pickedPath!.split(RegExp(r'[\\/]+')).last,
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
            ],
            if (_pickedBytes != null) ...[
              const SizedBox(height: 12),
              const Text('File loaded (web). Use an Android device/emulator to run on-device inference.'),
            ],
            const SizedBox(height: 28),
            const Text(
              'Tip: If the file is not 16kHz, the model expects 16k input. The TorchScript wrapper includes mel-preprocessing but still expects 16k sampling rate.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 12),
            ),
            const SizedBox(height: 24),
            const Text(
              'Note: On web the file path is unavailable; choose a local Android/iOS device for inference.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 12, color: Colors.black54),
            ),
          ],
        ),
      ),
    );
  }
}
