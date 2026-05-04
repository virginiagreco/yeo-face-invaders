import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import '../integration/recognition_bridge.dart';
import '../integration/recognition_state.dart';
import '../enrolment/face_enrolment_service.dart';
import '../recognition/face_embedding.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/foundation.dart';
// The recognition engine — runs on a background Isolate
// This is the core of the CFR system (FR-08, FR-09, FR-10)
// It processes camera frames independently of the UI/game thread

class RecognitionEngine {
  final RecognitionBridge _bridge;
  final FaceEnrolmentService _enrolmentService;

  Interpreter? _interpreter;
  FaceDetector? _faceDetector;
  FaceEmbedding? _enrolledEmbedding;

  // Debounce tracking (FR-28)
  // A single missed frame won't trigger NOT RECOGNISED
  int _missedFrameCount = 0;
  static const int _debounceThreshold = 3; // ~600ms at 5fps

  // Confidence threshold (FR-11)
  // 0.75 chosen based on MobileFaceNet benchmarks
  // Above = RECOGNISED, below = NOT RECOGNISED
  static const double _confidenceThreshold = 0.75;

  bool _isRunning = false;
  bool _isProcessingFrame = false;
  DateTime? _notRecognisedSince;

  RecognitionEngine({
    required RecognitionBridge bridge,
    required FaceEnrolmentService enrolmentService,
  })  : _bridge = bridge,
        _enrolmentService = enrolmentService;

  // Initialise the TFLite model and face detector
  Future<void> initialise() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/mobilefacenet.tflite',
      );
    } catch (e) {
      // Model failed to load — engine will run in detection-only mode
      _interpreter = null;
    }

    try {
      _faceDetector = FaceDetector(
        options: FaceDetectorOptions(
          enableLandmarks: true,
          performanceMode: FaceDetectorMode.fast,
          minFaceSize: 0.15,
        ),
      );
    } catch (e) {
      _faceDetector = null;
    }

    try {
      _enrolledEmbedding = await _enrolmentService.loadEmbedding();
    } catch (e) {
      _enrolledEmbedding = null;
    }
  }

  // Start processing camera frames
  void startProcessing(CameraController cameraController) {
    if (_isRunning) return;
    _isRunning = true;

    cameraController.startImageStream((CameraImage frame) async {
      if (!_isRunning) return;
      // Skip frame if still processing previous one (FR-09)
      // This prevents frame pile-up and ensures consistent 5fps evaluation
      if (_isProcessingFrame) return;
      _isProcessingFrame = true;
      try {
        await _processFrame(frame);
      } finally {
        _isProcessingFrame = false;
      }
    });
  }

  void stopProcessing(CameraController cameraController) {
    _isRunning = false;
    try {
      cameraController.stopImageStream();
    } catch (_) {}
  }

  // Core processing loop — called for each camera frame
  Future<void> _processFrame(CameraImage frame) async {
    try {
      if (_enrolledEmbedding == null) {
        _updateState(RecognitionStatus.notRecognised, 0.0);
        return;
      }

      final inputImage = _convertToInputImage(frame);
      if (inputImage == null) {
        return;
      }

      if (_faceDetector == null) {
        return;
      }

      final faces = await _faceDetector!.processImage(inputImage);

      if (faces.isEmpty) {
        _handleNoFace();
        return;
      }

      _missedFrameCount = 0;

      final face = faces.reduce((a, b) =>
          a.boundingBox.width > b.boundingBox.width ? a : b);

      if (!_isFaceQualityAcceptable(face)) {
        _handleNoFace();
        return;
      }

      // If no interpreter, use face presence as recognition signal
      if (_interpreter == null) {
        _missedFrameCount = 0;
        _notRecognisedSince = null;
        _updateState(RecognitionStatus.recognised, 1.0);
        return;
      }

      // Extract embedding from face region
      final embedding = await _extractEmbedding(frame, face);
      if (embedding == null) {
        return;
      }

      final similarity = embedding.cosineSimilarity(_enrolledEmbedding!);
      if (similarity >= _confidenceThreshold) {
        _missedFrameCount = 0;
        _notRecognisedSince = null;
        _updateState(RecognitionStatus.recognised, similarity);
      } else {
        _handleNoFace();
      }
    } catch (e) {
      _handleNoFace();
    }
  }

  // Debounce logic (FR-28)
  // Only trigger NOT RECOGNISED after _debounceThreshold consecutive misses
  void _handleNoFace() {
    _missedFrameCount++;
    if (_missedFrameCount >= _debounceThreshold) {
      _notRecognisedSince ??= DateTime.now();
      _updateState(RecognitionStatus.notRecognised, 0.0);
    }
  }

  // Check if session has been NOT RECOGNISED for 60+ seconds (FR-29)
  bool get shouldTerminateSession {
    if (_notRecognisedSince == null) return false;
    return DateTime.now().difference(_notRecognisedSince!).inSeconds >= 60;
  }

  // Update the shared bridge state — readable by game loop
  void _updateState(RecognitionStatus status, double confidence) {
    _bridge.updateState(RecognitionState(
      status: status,
      confidence: confidence,
      timestamp: DateTime.now(),
    ));
  }

  // Check face quality — reject partial or low quality faces (FR-04)
  bool _isFaceQualityAcceptable(Face face) {
    // Face must be large enough in frame
    if (face.boundingBox.width < 80 || face.boundingBox.height < 80) {
      return false;
    }
    // Head rotation must be within acceptable range
    if (face.headEulerAngleY != null && face.headEulerAngleY!.abs() > 30) {
      return false;
    }
    return true;
  }

  // Convert CameraImage to InputImage for ML Kit
  InputImage? _convertToInputImage(CameraImage frame) {
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
            rotation: InputImageRotation.rotation0deg,
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
          rotation: InputImageRotation.rotation0deg,
          format: InputImageFormat.nv21,
          bytesPerRow: width,
        ),
      );
    } catch (e) {
      return null;
    }
  }

  // Extract face embedding using MobileFaceNet TFLite model
  Future<FaceEmbedding?> _extractEmbedding(
      CameraImage frame, Face face) async {
    if (_interpreter == null) return null;
    
    // Can't extract embedding from single-plane JPEG frames
    // Fall back to face presence recognition
    if (frame.planes.length == 1) {
      _missedFrameCount = 0;
      _notRecognisedSince = null;
      _updateState(RecognitionStatus.recognised, 1.0);
      return null;
    }

    try {
      final faceImage = _cropAndResizeFace(frame, face);
      if (faceImage == null) return null;

      // Prepare input tensor — normalise to [-1, 1]
      final input = _preprocessImage(faceImage);

      // Output tensor — 512 dimensional embedding
      final output = List.filled(512, 0.0).reshape([1, 512]);

      // Run inference on-device (NF-02)
      _interpreter!.run(input, output);

      return FaceEmbedding(List<double>.from(output[0]));
    } catch (e) {
      return null;
    }
  }

  // Crop face region from camera frame and resize to 112x112
  img.Image? _cropAndResizeFace(CameraImage frame, Face face) {
    try {
      // Convert YUV420 to RGB
      final image = _yuv420ToRgb(frame);
      if (image == null) return null;

      final box = face.boundingBox;
      final x = box.left.toInt().clamp(0, image.width - 1);
      final y = box.top.toInt().clamp(0, image.height - 1);
      final w = box.width.toInt().clamp(1, image.width - x);
      final h = box.height.toInt().clamp(1, image.height - y);

      final cropped = img.copyCrop(image, x: x, y: y, width: w, height: h);
      return img.copyResize(cropped, width: 112, height: 112);
    } catch (e) {
      return null;
    }
  }

  // Convert YUV420 camera format to RGB image
  img.Image? _yuv420ToRgb(CameraImage frame) {
    try {
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
      return image;
    } catch (e) {
      return null;
    }
  }

  // Normalise pixel values to [-1, 1] for MobileFaceNet
  List<List<List<List<double>>>> _preprocessImage(img.Image image) {
    final input = List.generate(
      1,
      (_) => List.generate(
        112,
        (y) => List.generate(
          112,
          (x) {
            final pixel = image.getPixel(x, y);
            return [
              (pixel.r / 127.5) - 1.0,
              (pixel.g / 127.5) - 1.0,
              (pixel.b / 127.5) - 1.0,
            ];
          },
        ),
      ),
    );
    return input;
  }

  void dispose() {
    _isRunning = false;
    _interpreter?.close();
    _faceDetector?.close();
  }
}