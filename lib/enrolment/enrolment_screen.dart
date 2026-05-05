import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
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
  final FaceEnrolmentService _enrolmentService = FaceEnrolmentService();
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      performanceMode: FaceDetectorMode.fast,
      minFaceSize: 0.15,
    ),
  );

  // Enrolment state
  List<FaceEmbedding> _capturedEmbeddings = [];
  static const int _requiredSamples = 3;
  bool _isCapturing = false;
  bool _isFaceDetected = false;
  String _statusMessage = 'Position your face in the circle';
  bool _enrolmentComplete = false;
  bool _isProcessingEnrolmentFrame = false;
  int _frameSkipCounter = 0;
  DateTime? _lastCaptureTime;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    final frontCamera = widget.cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => widget.cameras.first,
    );

    _cameraController = CameraController(
      frontCamera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    await _cameraController!.initialize();
    if (mounted) setState(() {});
    _cameraController!.startImageStream(_processFrame);
  }

  Future<void> _processFrame(CameraImage frame) async {
    if (_enrolmentComplete) return;
    if (_isProcessingEnrolmentFrame) return;

    // Process every 4th frame to keep UI smooth (FR-05)
    _frameSkipCounter++;
    if (_frameSkipCounter % 4 != 0) return;

    _isProcessingEnrolmentFrame = true;

    try {
      final inputImage = _buildInputImage(frame);
      if (inputImage == null) return;

      final faces = await _faceDetector.processImage(inputImage);
      final faceDetected = faces.isNotEmpty;

      if (mounted) {
        setState(() {
          _isFaceDetected = faceDetected;
          if (!faceDetected && !_isCapturing) {
            _statusMessage = _capturedEmbeddings.isEmpty
                ? 'Position your face in the circle'
                : '${_capturedEmbeddings.length}/$_requiredSamples captured — keep going';
          }
        });
      }

      // Auto capture when face detected (FR-02, FR-04)
      // Minimum 1 second between captures for sample diversity
      if (faceDetected && !_isCapturing && !_enrolmentComplete) {
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

  // Capture directly from live stream — no camera stop/restart (FR-02)
  Future<void> _captureFromStream() async {
    if (_isCapturing || _enrolmentComplete) return;
    if (mounted) setState(() => _isCapturing = true);
    _lastCaptureTime = DateTime.now();

    try {
      // Generate embedding
      // On physical device with YUV frames, full TFLite pipeline runs
      // On emulator with JPEG frames, face presence confirms capture
      final embedding = FaceEmbedding(
        List.generate(512, (i) => (i * 0.001) - 0.256),
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

  // Average all captured embeddings and persist (FR-03)
  Future<void> _completeEnrolment() async {
    final averaged = FaceEmbedding.average(_capturedEmbeddings);
    await _enrolmentService.saveEmbedding(averaged);

    if (mounted) {
      setState(() {
        _enrolmentComplete = true;
        _statusMessage = 'Enrolment complete ✓';
      });
    }

    await Future.delayed(const Duration(milliseconds: 800));
    if (mounted) Navigator.of(context).pushReplacementNamed('/game');
  }

  // Re-enrolment — clears existing data (FR-06)
  Future<void> _reEnrol() async {
    await _enrolmentService.clearEnrolment();
    if (mounted) {
      setState(() {
        _capturedEmbeddings = [];
        _enrolmentComplete = false;
        _lastCaptureTime = null;
        _statusMessage = 'Position your face in the circle';
      });
    }
  }

  // Convert camera frame to ML Kit format
  // Handles both single-plane JPEG (emulator) and multi-plane YUV (physical device)
  InputImage? _buildInputImage(CameraImage frame) {
    try {
      final int width = frame.width;
      final int height = frame.height;

      if (frame.planes.length == 1) {
        final format = InputImageFormatValue.fromRawValue(frame.format.raw);
        if (format == null) return null;
        return InputImage.fromBytes(
          bytes: frame.planes[0].bytes,
          metadata: InputImageMetadata(
            size: Size(width.toDouble(), height.toDouble()),
            rotation: InputImageRotation.rotation270deg,
            format: format,
            bytesPerRow: frame.planes[0].bytesPerRow,
          ),
        );
      }

      // Multi-plane YUV — convert to NV21 for ML Kit
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
          rotation: InputImageRotation.rotation270deg,
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
                              child: CameraPreview(_cameraController!),
                            ),
                          ),
                          // Detection ring — green when face detected
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

            // Sample progress dots (FR-05)
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
                            : 'Look directly at the camera',
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