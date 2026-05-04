import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'face_enrolment_service.dart';
import '../recognition/face_embedding.dart';
import 'package:flutter/foundation.dart';

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

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    // Use front camera for enrolment (FR-01)
    final frontCamera = widget.cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => widget.cameras.first,
    );

    _cameraController = CameraController(
      frontCamera,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await _cameraController!.initialize();
    if (mounted) setState(() {});

    // Start face detection stream for real-time feedback (FR-05)
    _cameraController!.startImageStream(_detectFaceInFrame);
  }

  // Real-time face detection for visual feedback
  Future<void> _detectFaceInFrame(CameraImage frame) async {
    if (_isCapturing || _enrolmentComplete) return;
    if (_isProcessingEnrolmentFrame) return;
    _isProcessingEnrolmentFrame = true;

    try {
      final inputImage = _buildInputImage(frame);
      if (inputImage == null) return;

      final faces = await _faceDetector.processImage(inputImage);
      if (mounted) {
        setState(() {
          _isFaceDetected = faces.isNotEmpty;
          _statusMessage = faces.isNotEmpty
              ? 'Face detected — tap Capture'
              : 'Position your face in the circle';
        });
      }
    } catch (_) {
    } finally {
      _isProcessingEnrolmentFrame = false;
    }
  }

  // Capture a single face sample
  Future<void> _captureSample() async {
    if (_isCapturing || !_isFaceDetected) return;
    setState(() {
      _isCapturing = true;
      _statusMessage = 'Capturing...';
    });

    try {
      await _cameraController!.stopImageStream();
      final image = await _cameraController!.takePicture();

      // Build embedding from captured image
      final embedding = await _buildEmbeddingFromPath(image.path);

      if (embedding != null) {
        setState(() {
          _capturedEmbeddings.add(embedding);
          _statusMessage =
              '${_capturedEmbeddings.length}/$_requiredSamples samples captured';
        });

        if (_capturedEmbeddings.length >= _requiredSamples) {
          await _completeEnrolment();
        } else {
          // Restart stream for next capture
          await _cameraController!.startImageStream(_detectFaceInFrame);
        }
      }
    } catch (e) {
      setState(() => _statusMessage = 'Capture failed — try again');
      await _cameraController!.startImageStream(_detectFaceInFrame);
    } finally {
      setState(() => _isCapturing = false);
    }
  }

  // Average all captured embeddings and save (FR-02, FR-03)
  Future<void> _completeEnrolment() async {
    final averaged = FaceEmbedding.average(_capturedEmbeddings);
    await _enrolmentService.saveEmbedding(averaged);
    setState(() {
      _enrolmentComplete = true;
      _statusMessage = 'Enrolment complete!';
    });

    // Stop camera stream and dispose before navigating
    try {
      await _cameraController?.stopImageStream();
    } catch (_) {}
    
    await Future.delayed(const Duration(milliseconds: 500));
    
    if (mounted) {
      Navigator.of(context).pushReplacementNamed('/game');
    }
  }

  // Re-enrolment — clear existing and start again (FR-06)
  Future<void> _reEnrol() async {
    await _enrolmentService.clearEnrolment();
    setState(() {
      _capturedEmbeddings = [];
      _enrolmentComplete = false;
      _statusMessage = 'Position your face in the circle';
    });
    await _cameraController!.startImageStream(_detectFaceInFrame);
  }

  InputImage? _buildInputImage(CameraImage frame) {
    try {
      final int width = frame.width;
      final int height = frame.height;

      if (frame.planes.length == 1) {
        // Single plane — detect actual format
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

      // Multi-plane YUV — convert to NV21
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
    } catch (e) {
      return null;
    }
  }

  Future<FaceEmbedding?> _buildEmbeddingFromPath(String path) async {
    // Placeholder — real embedding extracted by recognition engine
    // Returns a dummy embedding for now, replaced in Day 2
    return FaceEmbedding(List.generate(512, (i) => i * 0.001));
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

            // Camera preview with face overlay
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
                          // Face detection indicator ring
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

            // Sample progress indicators
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: List.generate(_requiredSamples, (i) {
                  return Container(
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

            // Capture button
            Padding(
              padding: const EdgeInsets.only(bottom: 32),
              child: Column(
                children: [
                  ElevatedButton(
                    onPressed:
                        _isFaceDetected && !_isCapturing && !_enrolmentComplete
                            ? _captureSample
                            : null,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.greenAccent,
                      foregroundColor: Colors.black,
                      padding: const EdgeInsets.symmetric(
                          horizontal: 48, vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(32),
                      ),
                    ),
                    child: Text(
                      _isCapturing ? 'Capturing...' : 'Capture Sample',
                      style: const TextStyle(
                          fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                  ),
                  const SizedBox(height: 12),
                  TextButton(
                    onPressed: _reEnrol,
                    child: const Text(
                      'Re-enrol',
                      style: TextStyle(color: Colors.white54),
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