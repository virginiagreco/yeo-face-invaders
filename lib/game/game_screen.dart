import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:camera/camera.dart';
import '../integration/recognition_bridge.dart';
import '../recognition/recognition_engine.dart';
import '../enrolment/face_enrolment_service.dart';
import 'space_invaders_game.dart';
import 'package:permission_handler/permission_handler.dart';

class GameScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const GameScreen({super.key, required this.cameras});

  @override
  State<GameScreen> createState() => _GameScreenState();
}

class _GameScreenState extends State<GameScreen>
    with SingleTickerProviderStateMixin {
  // Game
  final SpaceInvadersGame _game = SpaceInvadersGame();
  Ticker? _ticker;
  Duration _lastElapsed = Duration.zero;

  // Recognition
  late RecognitionEngine _recognitionEngine;
  late RecognitionBridge _bridge;
  CameraController? _cameraController;

  // State
  bool _isRecognised = false;
  double _playerMoveDirection = 0.0;

  @override
  void initState() {
    super.initState();
    _bridge = RecognitionBridge();
   // _isRecognised = true; // Bypass for emulator testing
    _recognitionEngine = RecognitionEngine(
      bridge: _bridge,
      enrolmentService: FaceEnrolmentService(),
    );

    // Listen to recognition state changes (FR-23 to FR-26)
    _bridge.stream.listen((state) {
      if (!mounted) return;
      setState(() => _isRecognised = state.isRecognised);

      // Check 60 second timeout (FR-29)
      if (_recognitionEngine.shouldTerminateSession) {
        _terminateSession();
      }
    });

    _initRecognition();

    // Game ticker — drives the game loop at 60fps (FR-21)
    _ticker = createTicker(_onTick)..start();
  }

    Future<void> _initRecognition() async {
    try {
    // Request camera permission (NF-06)
    final status = await Permission.camera.request();
    if (!status.isGranted) return;
      // Wait for enrolment camera to fully release
      await Future.delayed(const Duration(milliseconds: 500));
      
      await _recognitionEngine.initialise();

      final frontCamera = widget.cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => widget.cameras.first,
      );

      _cameraController = CameraController(
        frontCamera,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _cameraController!.initialize();
      _recognitionEngine.startProcessing(_cameraController!);
    } catch (e) {
      // Camera stop errors ignored on dispose      
    }
  }

  // Game loop — called every frame by the Ticker (FR-22)
  void _onTick(Duration elapsed) {
    if (_lastElapsed == Duration.zero) {
      _lastElapsed = elapsed;
      return;
    }

    final dt = (elapsed - _lastElapsed).inMicroseconds / 1000000.0;
    _lastElapsed = elapsed;

    // Only update game if recognised (FR-23)
    if (_isRecognised && !_game.gameOver) {
      _game.update(dt);
      _game.movePlayer(_playerMoveDirection, dt);
    }

    setState(() {}); // Trigger repaint
  }

  // Session terminated after 60s unrecognised (FR-29)
  void _terminateSession() {
    if (_cameraController != null) {
      try {
        _recognitionEngine.stopProcessing(_cameraController!);
      } catch (_) {}
    }
    Navigator.of(context).pushReplacementNamed('/enrolment');
  }

  @override
  void dispose() {
    _ticker?.dispose();
    if (_cameraController != null) {
      try {
        _recognitionEngine.stopProcessing(_cameraController!);
      } catch (_) {}
    }
    _recognitionEngine.dispose();
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            // Initialise game with screen dimensions
            if (_game.screenWidth == 0) {
              _game.initialise(constraints.maxWidth, constraints.maxHeight - 80);
            }

            // FR-24, FR-25: Blank screen when not recognised
            if (!_isRecognised) {
              return _buildPausedScreen();
            }

            return _buildGameScreen(constraints);
          },
        ),
      ),
    );
  }

  // Blanked screen — shown when face not detected (FR-24, FR-25)
  Widget _buildPausedScreen() {
    return Container(
      color: Colors.black,
      child: const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.face, color: Colors.white24, size: 64),
            SizedBox(height: 16),
            Text(
              'Face not detected — game paused',
              style: TextStyle(color: Colors.white54, fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildGameScreen(BoxConstraints constraints) {
    return Column(
      children: [
        // Game canvas
        Expanded(
          child: GestureDetector(
            onTapDown: (_) => _game.shoot(),
            child: CustomPaint(
              painter: SpaceInvadersPainter(_game),
              size: Size(constraints.maxWidth, constraints.maxHeight - 80),
            ),
          ),
        ),

        // Touch controls (FR-16)
        SizedBox(
          height: 80,
          child: Row(
            children: [
              // Move left
              Expanded(
                child: GestureDetector(
                  onTapDown: (_) => _playerMoveDirection = -1.0,
                  onTapUp: (_) => _playerMoveDirection = 0.0,
                  onTapCancel: () => _playerMoveDirection = 0.0,
                  child: Container(
                    color: Colors.white10,
                    child: const Icon(Icons.arrow_left,
                        color: Colors.white54, size: 48),
                  ),
                ),
              ),

              // Fire button
              GestureDetector(
                onTap: () => _game.shoot(),
                child: Container(
                  width: 80,
                  color: Colors.white10,
                  child: const Icon(Icons.circle,
                      color: Colors.redAccent, size: 40),
                ),
              ),

              // Move right
              Expanded(
                child: GestureDetector(
                  onTapDown: (_) => _playerMoveDirection = 1.0,
                  onTapUp: (_) => _playerMoveDirection = 0.0,
                  onTapCancel: () => _playerMoveDirection = 0.0,
                  child: Container(
                    color: Colors.white10,
                    child: const Icon(Icons.arrow_right,
                        color: Colors.white54, size: 48),
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}