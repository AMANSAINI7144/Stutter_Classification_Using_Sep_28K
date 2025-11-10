// lib/main.dart
import 'package:flutter/material.dart';
import 'pages/choose_file_page.dart';
import 'pages/buffering_page.dart';
import 'pages/result_page.dart';

void main() {
  runApp(const StutterApp());
}

class StutterApp extends StatelessWidget {
  const StutterApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Stutter Analyzer',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
        scaffoldBackgroundColor: const Color(0xFFFFF8FB),
        useMaterial3: false,
      ),
      // The first screen the app opens on
      initialRoute: '/',
      onGenerateRoute: (settings) {
        switch (settings.name) {
          case '/':
            return MaterialPageRoute(builder: (_) => const ChooseFilePage());
          case '/buffering':
            final args = settings.arguments as Map<String, dynamic>?;
            final wavPath = args?['wavPath'] ?? '';
            return MaterialPageRoute(
              builder: (_) => BufferingPage(wavPath: wavPath),
            );
          case '/result':
            final args = settings.arguments as Map<String, dynamic>?;
            final probs = (args?['probabilities'] as List<dynamic>?)
                ?.map((e) => (e as num).toDouble())
                .toList() ??
                [];
            return MaterialPageRoute(
              builder: (_) => ResultPage(probabilities: probs),
            );
          default:
            return MaterialPageRoute(
              builder: (_) => const ChooseFilePage(),
            );
        }
      },
    );
  }
}
