// lib/pages/result_page.dart
import 'package:flutter/material.dart';

class ResultPage extends StatelessWidget {
  final List<double> probabilities;
  const ResultPage({super.key, required this.probabilities});

  @override
  Widget build(BuildContext context) {
    // Simple class labels — adjust as your model's classes
    final classLabels = [
      'block',
      'prolong',
      'revise',
      'part_word',
      'interjection',
      'fluent' // example names — replace with your real labels
    ];

    return Scaffold(
      appBar: AppBar(
        title: const Text('Result'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
        child: Column(
          children: [
            const Text('Model output probabilities:', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 14),
            Expanded(
              child: ListView.separated(
                itemCount: probabilities.length,
                separatorBuilder: (_, __) => const Divider(),
                itemBuilder: (context, idx) {
                  final label = idx < classLabels.length ? classLabels[idx] : 'class_$idx';
                  final prob = probabilities[idx];
                  final pct = (prob * 100.0);
                  return ListTile(
                    title: Text(label),
                    subtitle: LinearProgressIndicator(value: prob.clamp(0.0, 1.0)),
                    trailing: Text(pct.toStringAsFixed(1) + '%'),
                  );
                },
              ),
            ),
            const SizedBox(height: 12),
            ElevatedButton.icon(
              onPressed: () {
                // Pop back to root (choose file)
                Navigator.popUntil(context, (route) => route.isFirst);
              },
              icon: const Icon(Icons.arrow_back),
              label: const Text('Back to choose file'),
            ),
          ],
        ),
      ),
    );
  }
}
