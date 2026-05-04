import 'dart:async';
import 'recognition_state.dart';

// The bridge exposes a stream that both the recognition engine
// and the game loop can access — this is how they communicate
// without blocking each other

class RecognitionBridge {
  // Singleton — only one instance exists in the whole app
  static final RecognitionBridge _instance = RecognitionBridge._internal();
  factory RecognitionBridge() => _instance;
  RecognitionBridge._internal();

  // StreamController broadcasts state changes to anyone listening
  final _controller = StreamController<RecognitionState>.broadcast();

  // The game listens to this stream
  Stream<RecognitionState> get stream => _controller.stream;

  // Current state — readable by game loop on each tick
  RecognitionState _current = RecognitionState.initial;
  RecognitionState get current => _current;

  // Called by the recognition engine when state changes
  void updateState(RecognitionState newState) {
    _current = newState;
    if (!_controller.isClosed) {
      _controller.add(newState);
    }
  }

  void dispose() {
    _controller.close();
  }
}