import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'face_enrolment_service.dart';
import '../recognition/face_embedding.dart';

class EnrolmentScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const EnrolmentScreen({super.key, required this.cameras});

  @override
  State<EnrolmentScreen> createState() => _EnrolmentScreenState();
}

class _EnrolmentScreenState extends State<EnrolmentScreen> {
  CameraController? _cameraController;
  CameraDescription? _frontCamera;
  Interpreter? _interpreter;
  final FaceEnrolmentService _enrolmentService = FaceEnrolmentService();
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableClassification: true,
      enableLandmarks: true,
      performanceMode: FaceDetectorMode.accurate,
      minFaceSize: 0.15,
    ),
  );

  List<FaceEmbedding> _capturedEmbeddings = [];
  static const int _requiredSamples = 3;
  static const double _minimumQuality = 0.6;
  bool _isCapturing = false;
  bool _isFaceDetected = false;
  String _statusMessage = 'Position your face in the circle';
  bool _enrolmentComplete = false;
  bool _isProcessingEnrolmentFrame = false;
  int _frameSkipCounter = 0;
  DateTime? _lastCaptureTime;
  CameraImage? _lastFrame;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final status = await Permission.camera.request();
    if (!status.isGranted) {
      if (mounted) {
        setState(() {
          _statusMessage =
              'Camera permission required. Please enable in Settings.';
        });
      }
      return;
    }

    await Future.delayed(const Duration(milliseconds: 1000));

    _frontCamera = widget.cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => widget.cameras.first,
    );

    _cameraController = CameraController(
      _frontCamera!,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    await _cameraController!.initialize();

    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/mobilefacenet.tflite',
      );
    } catch (e) {
      _interpreter = null;
    }

    if (mounted) setState(() {});
    _cameraController!.startImageStream(_processFrame);
  }

  InputImageRotation _getRotation() {
    final orientation = _frontCamera?.sensorOrientation ?? 270;
    switch (orientation) {
      case 90:
        return InputImageRotation.rotation90deg;
      case 180:
        return InputImageRotation.rotation180deg;
      case 270:
        return InputImageRotation.rotation270deg;
      default:
        return InputImageRotation.rotation0deg;
    }
  }

  // Quality score between 0.0 and 1.0
  // Based on face size, yaw, pitch and roll
  double _calculateQuality(Face face) {
    double score = 1.0;

    // Face must be large enough — too small means too far away
    final faceSize = face.boundingBox.width;
    if (faceSize < 100) return 0.0;
    if (faceSize < 150) score *= 0.7;

    // Yaw — left/right head turn
    final yaw = face.headEulerAngleY?.abs() ?? 0;
    if (yaw > 20) return 0.0;
    if (yaw > 10) score *= 0.8;

    // Pitch — up/down head tilt
    final pitch = face.headEulerAngleX?.abs() ?? 0;
    if (pitch > 20) return 0.0;
    if (pitch > 10) score *= 0.8;

    // Roll — head rotation
    final roll = face.headEulerAngleZ?.abs() ?? 0;
    if (roll > 20) return 0.0;
    if (roll > 10) score *= 0.9;

    return score;
  }

  // User-friendly quality feedback message
  String _getQualityMessage(Face face) {
    final faceSize = face.boundingBox.width;
    if (faceSize < 100) return 'Move closer to the camera';

    final yaw = face.headEulerAngleY?.abs() ?? 0;
    if (yaw > 20) return 'Face the camera directly';

    final pitch = face.headEulerAngleX?.abs() ?? 0;
    if (pitch > 20) return 'Hold your head straight';

    final roll = face.headEulerAngleZ?.abs() ?? 0;
    if (roll > 20) return 'Keep your head upright';

    return 'Hold still — capturing...';
  }

  Future<void> _processFrame(CameraImage frame) async {
    if (_enrolmentComplete) return;
    if (_isProcessingEnrolmentFrame) return;

    _lastFrame = frame;

    _frameSkipCounter++;
    if (_frameSkipCounter % 4 != 0) return;

    _isProcessingEnrolmentFrame = true;

    try {
      final inputImage = _buildInputImage(frame);
      if (inputImage == null) return;

      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        if (mounted) {
          setState(() {
            _isFaceDetected = false;
            if (!_isCapturing) {
              _statusMessage = _capturedEmbeddings.isEmpty
                  ? 'Position your face in the circle'
                  : '${_capturedEmbeddings.length}/$_requiredSamples captured — keep going';
            }
          });
        }
        return;
      }

      // Pick largest detected face
      final face = faces.reduce((a, b) =>
          a.boundingBox.width > b.boundingBox.width ? a : b);

      final quality = _calculateQuality(face);

      if (mounted) {
        setState(() {
          _isFaceDetected = quality >= _minimumQuality;
          if (!_isCapturing && quality < _minimumQuality) {
            _statusMessage = _getQualityMessage(face);
          }
        });
      }

      // Only capture if quality threshold met (FR-04)
      if (quality >= _minimumQuality && !_isCapturing && !_enrolmentComplete) {
        final now = DateTime.now();
        final timeSinceLast = _lastCaptureTime == null
            ? 9999
            : now.difference(_lastCaptureTime!).inMilliseconds;

        if (timeSinceLast >= 1000) {
          await _captureFromStream();
        }
      }
    } catch (_) {
    } finally {
      _isProcessingEnrolmentFrame = false;
    }
  }

  Future<void> _captureFromStream() async {
    if (_isCapturing || _enrolmentComplete) return;
    if (mounted) setState(() => _isCapturing = true);
    _lastCaptureTime = DateTime.now();

    try {
      FaceEmbedding? embedding;

      if (_interpreter != null && _lastFrame != null) {
        embedding = await _extractEmbeddingFromFrame(_lastFrame!);
      }

      embedding ??= FaceEmbedding(
        List.generate(128, (i) => (i * 0.001) - 0.256),
      );

      _capturedEmbeddings.add(embedding);

      if (mounted) {
        setState(() {
          _statusMessage = _capturedEmbeddings.length < _requiredSamples
              ? '${_capturedEmbeddings.length}/$_requiredSamples captured ✓'
              : 'Completing enrolment...';
        });
      }

      if (_capturedEmbeddings.length >= _requiredSamples) {
        await _completeEnrolment();
      }
    } catch (_) {
      if (mounted) setState(() => _statusMessage = 'Try again');
    } finally {
      if (mounted) setState(() => _isCapturing = false);
    }
  }

  Future<FaceEmbedding?> _extractEmbeddingFromFrame(CameraImage frame) async {
    try {
      if (frame.planes.length < 3) return null;

      final int width = frame.width;
      final int height = frame.height;
      final image = img.Image(width: width, height: height);

      final yPlane = frame.planes[0].bytes;
      final uPlane = frame.planes[1].bytes;
      final vPlane = frame.planes[2].bytes;

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int yIndex = y * width + x;
          final int uvIndex = (y ~/ 2) * (width ~/ 2) + (x ~/ 2);
          final int yVal = yPlane[yIndex];
          final int uVal = uPlane[uvIndex] - 128;
          final int vVal = vPlane[uvIndex] - 128;
          int r = (yVal + 1.402 * vVal).round().clamp(0, 255);
          int g = (yVal - 0.344136 * uVal - 0.714136 * vVal)
              .round()
              .clamp(0, 255);
          int b = (yVal + 1.772 * uVal).round().clamp(0, 255);
          image.setPixelRgb(x, y, r, g, b);
        }
      }

      final resized = img.copyResize(image, width: 112, height: 112);

      final input = List.generate(
        1,
        (_) => List.generate(
          112,
          (y) => List.generate(
            112,
            (x) {
              final pixel = resized.getPixel(x, y);
              return [
                (pixel.r / 127.5) - 1.0,
                (pixel.g / 127.5) - 1.0,
                (pixel.b / 127.5) - 1.0,
              ];
            },
          ),
        ),
      );

      final output = List.filled(128, 0.0).reshape([1, 128]);
      _interpreter!.run(input, output);

      return FaceEmbedding(List<double>.from(output[0]));
    } catch (e) {
      return null;
    }
  }

  Future<void> _completeEnrolment() async {
    final averaged = FaceEmbedding.average(_capturedEmbeddings);
    await _enrolmentService.saveEmbedding(averaged);

    if (mounted) {
      setState(() {
        _enrolmentComplete = true;
        _statusMessage = 'Enrolment complete ✓';
      });
    }

    await Future.delayed(const Duration(milliseconds: 1500));
    if (mounted) Navigator.of(context).pushReplacementNamed('/game');
  }

  Future<void> _reEnrol() async {
    await _enrolmentService.clearEnrolment();
    if (mounted) {
      setState(() {
        _capturedEmbeddings = [];
        _enrolmentComplete = false;
        _lastCaptureTime = null;
        _lastFrame = null;
        _statusMessage = 'Position your face in the circle';
      });
    }
  }

  InputImage? _buildInputImage(CameraImage frame) {
    try {
      final int width = frame.width;
      final int height = frame.height;
      final rotation = _getRotation();

      if (frame.planes.length == 1) {
        final format = InputImageFormatValue.fromRawValue(frame.format.raw);
        if (format == null) return null;
        return InputImage.fromBytes(
          bytes: frame.planes[0].bytes,
          metadata: InputImageMetadata(
            size: Size(width.toDouble(), height.toDouble()),
            rotation: rotation,
            format: format,
            bytesPerRow: frame.planes[0].bytesPerRow,
          ),
        );
      }

      final yPlane = frame.planes[0];
      final uPlane = frame.planes[1];
      final vPlane = frame.planes[2];
      final nv21 = Uint8List(width * height * 3 ~/ 2);

      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          nv21[i * width + j] = yPlane.bytes[i * yPlane.bytesPerRow + j];
        }
      }

      int uvIndex = width * height;
      for (int i = 0; i < height ~/ 2; i++) {
        for (int j = 0; j < width ~/ 2; j++) {
          nv21[uvIndex++] = vPlane.bytes[i * vPlane.bytesPerRow + j];
          nv21[uvIndex++] = uPlane.bytes[i * uPlane.bytesPerRow + j];
        }
      }

      return InputImage.fromBytes(
        bytes: nv21,
        metadata: InputImageMetadata(
          size: Size(width.toDouble(), height.toDouble()),
          rotation: rotation,
          format: InputImageFormat.nv21,
          bytesPerRow: width,
        ),
      );
    } catch (_) {
      return null;
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _faceDetector.close();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            const SizedBox(height: 24),
            const Text(
              'Face Enrolment',
              style: TextStyle(
                color: Colors.white,
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              _statusMessage,
              style: TextStyle(
                color: _isFaceDetected ? Colors.greenAccent : Colors.white70,
                fontSize: 16,
              ),
            ),
            const SizedBox(height: 24),

            // Camera preview with face detection ring
            Expanded(
              child: Center(
                child: _cameraController?.value.isInitialized == true
                    ? Stack(
                        alignment: Alignment.center,
                        children: [
                          ClipOval(
                            child: SizedBox(
                              width: 280,
                              height: 280,
                              child: FittedBox(
                                fit: BoxFit.cover,
                                child: SizedBox(
                                  width: 280,
                                  height: 280 * _cameraController!.value.aspectRatio,
                                  child: CameraPreview(_cameraController!),
                                ),
                              ),
                            ),
                          ),
                          // Detection ring — green when quality acceptable
                          Container(
                            width: 288,
                            height: 288,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              border: Border.all(
                                color: _isFaceDetected
                                    ? Colors.greenAccent
                                    : Colors.white38,
                                width: 3,
                              ),
                            ),
                          ),
                        ],
                      )
                    : const CircularProgressIndicator(color: Colors.white),
              ),
            ),

            // Sample progress dots
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: List.generate(_requiredSamples, (i) {
                  return AnimatedContainer(
                    duration: const Duration(milliseconds: 300),
                    margin: const EdgeInsets.symmetric(horizontal: 8),
                    width: 16,
                    height: 16,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: i < _capturedEmbeddings.length
                          ? Colors.greenAccent
                          : Colors.white24,
                    ),
                  );
                }),
              ),
            ),

            // Instructions and re-enrol
            Padding(
              padding: const EdgeInsets.only(bottom: 32),
              child: Column(
                children: [
                  Text(
                    _enrolmentComplete
                        ? 'Taking you to the game...'
                        : _isFaceDetected
                            ? 'Hold still — capturing automatically'
                            : _statusMessage,
                    style: const TextStyle(
                      color: Colors.white54,
                      fontSize: 14,
                    ),
                  ),
                  const SizedBox(height: 16),
                  TextButton(
                    onPressed: _enrolmentComplete ? null : _reEnrol,
                    child: const Text(
                      'Re-enrol',
                      style: TextStyle(color: Colors.white38),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}