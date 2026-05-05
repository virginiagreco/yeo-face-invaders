import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import '../integration/recognition_bridge.dart';
import '../recognition/recognition_engine.dart';
import '../enrolment/face_enrolment_service.dart';
import 'space_invaders_game.dart';

class GameScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const GameScreen({super.key, required this.cameras});

  @override
  State<GameScreen> createState() => _GameScreenState();
}

class _GameScreenState extends State<GameScreen>
    with SingleTickerProviderStateMixin {
  final SpaceInvadersGame _game = SpaceInvadersGame();
  Ticker? _ticker;
  Duration _lastElapsed = Duration.zero;

  late RecognitionEngine _recognitionEngine;
  late RecognitionBridge _bridge;
  CameraController? _cameraController;

  bool _isRecognised = false;
  double _playerMoveDirection = 0.0;

  @override
  void initState() {
    super.initState();
    _bridge = RecognitionBridge();
    _recognitionEngine = RecognitionEngine(
      bridge: _bridge,
      enrolmentService: FaceEnrolmentService(),
    );

    _bridge.stream.listen((state) {
      if (!mounted) return;
      setState(() => _isRecognised = state.isRecognised);

      if (_recognitionEngine.shouldTerminateSession) {
        _terminateSession();
      }
    });

    _initRecognition();

    _ticker = createTicker(_onTick)..start();
  }

  Future<void> _initRecognition() async {
    try {
      final status = await Permission.camera.request();
      if (!status.isGranted) return;

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
      _recognitionEngine.startProcessing(_cameraController!, frontCamera);
    } catch (e) {
      // Camera errors ignored on dispose
    }
  }

  void _onTick(Duration elapsed) {
    if (_lastElapsed == Duration.zero) {
      _lastElapsed = elapsed;
      return;
    }

    final dt = (elapsed - _lastElapsed).inMicroseconds / 1000000.0;
    _lastElapsed = elapsed;

    if (_isRecognised && !_game.gameOver) {
      _game.update(dt);
      _game.movePlayer(_playerMoveDirection, dt);
    }

    setState(() {});
  }

  void _terminateSession() async {
    _ticker?.stop();
    if (_cameraController != null) {
      try {
        _recognitionEngine.stopProcessing(_cameraController!);
        await _cameraController!.stopImageStream();
        await _cameraController!.dispose();
        _cameraController = null;
      } catch (_) {}
    }
    await Future.delayed(const Duration(milliseconds: 1500));
    if (mounted) {
      Navigator.of(context).pushReplacementNamed('/enrolment');
    }
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
            if (_game.screenWidth == 0) {
              _game.initialise(
                  constraints.maxWidth, constraints.maxHeight - 80);
            }

            if (!_isRecognised) {
              return _buildPausedScreen();
            }

            return _buildGameScreen(constraints);
          },
        ),
      ),
    );
  }

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
        Expanded(
          child: GestureDetector(
            onTapDown: (_) {
              if (_game.gameOver) {
                setState(() => _game.reset());
              } else {
                _game.shoot();
              }
            },
            child: CustomPaint(
              painter: SpaceInvadersPainter(_game),
              size: Size(constraints.maxWidth, constraints.maxHeight - 80),
            ),
          ),
        ),
        SizedBox(
          height: 80,
          child: Row(
            children: [
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
              GestureDetector(
                onTap: () => _game.shoot(),
                child: Container(
                  width: 80,
                  color: Colors.white10,
                  child: const Icon(Icons.circle,
                      color: Colors.redAccent, size: 40),
                ),
              ),
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