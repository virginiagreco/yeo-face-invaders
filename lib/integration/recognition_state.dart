// Integration layer — shared recognition state between engine and game
// This is the bridge that connects the two systems thread-safely

enum RecognitionStatus { recognised, notRecognised }

class RecognitionState {
  final RecognitionStatus status;
  final double confidence;
  final DateTime timestamp;

  const RecognitionState({
    required this.status,
    required this.confidence,
    required this.timestamp,
  });

  bool get isRecognised => status == RecognitionStatus.recognised;

  static RecognitionState get initial => RecognitionState(
        status: RecognitionStatus.notRecognised,
        confidence: 0.0,
        timestamp: DateTime.now(),
      );
}