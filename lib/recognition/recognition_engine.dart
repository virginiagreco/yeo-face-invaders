import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import '../integration/recognition_bridge.dart';
import '../integration/recognition_state.dart';
import '../enrolment/face_enrolment_service.dart';
import '../recognition/face_embedding.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class RecognitionEngine {
  final RecognitionBridge _bridge;
  final FaceEnrolmentService _enrolmentService;

  Interpreter? _interpreter;
  FaceDetector? _faceDetector;
  FaceEmbedding? _enrolledEmbedding;
  CameraDescription? _cameraDescription;

  int _missedFrameCount = 0;
  static const int _debounceThreshold = 3;
  static const double _confidenceThreshold = 0.75;

  bool _isRunning = false;
  bool _isProcessingFrame = false;
  DateTime? _notRecognisedSince;
  DateTime? _faceLastSeen;

  RecognitionEngine({
    required RecognitionBridge bridge,
    required FaceEnrolmentService enrolmentService,
  })  : _bridge = bridge,
        _enrolmentService = enrolmentService;

  Future<void> initialise() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/mobilefacenet.tflite',
      );
    } catch (e) {
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

  InputImageRotation _getRotation() {
    final orientation = _cameraDescription?.sensorOrientation ?? 270;
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

  void startProcessing(
    CameraController cameraController,
    CameraDescription cameraDescription,
  ) {
    _cameraDescription = cameraDescription;
    if (_isRunning) return;
    _isRunning = true;

    cameraController.startImageStream((CameraImage frame) async {
      if (!_isRunning) return;
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

  Future<void> _processFrame(CameraImage frame) async {
    try {
      if (_enrolledEmbedding == null) {
        _updateState(RecognitionStatus.notRecognised, 0.0);
        return;
      }

      final inputImage = _convertToInputImage(frame);
      if (inputImage == null) return;

      if (_faceDetector == null) return;

      final faces = await _faceDetector!.processImage(inputImage);

      if (faces.isEmpty) {
        _handleNoFace();
        return;
      }

      // Face is present — reset missed frame tracking
      _faceLastSeen ??= DateTime.now();
      _missedFrameCount = 0;

      final face = faces.reduce((a, b) =>
          a.boundingBox.width > b.boundingBox.width ? a : b);

      if (!_isFaceQualityAcceptable(face)) {
        _handleNoFace();
        return;
      }

      if (frame.planes.length == 1) {
        _faceLastSeen = DateTime.now();
        _missedFrameCount = 0;
        _notRecognisedSince = null;
        _updateState(RecognitionStatus.recognised, 1.0);
        return;
      }

      if (_interpreter == null) {
        _faceLastSeen = DateTime.now();
        _missedFrameCount = 0;
        _notRecognisedSince = null;
        _updateState(RecognitionStatus.recognised, 1.0);
        return;
      }

      final embedding = await _extractEmbedding(frame, face);
      if (embedding == null) return;

      final similarity = embedding.cosineSimilarity(_enrolledEmbedding!);

      if (similarity >= _confidenceThreshold) {
        _faceLastSeen = DateTime.now();
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

  // Remove _faceLastSeen ??= DateTime.now() from _handleNoFace
  void _handleNoFace() {
    _missedFrameCount++;
    if (_missedFrameCount >= _debounceThreshold) {
      if (_faceLastSeen != null) {
        _faceLastSeen = null;
      }
      _notRecognisedSince ??= DateTime.now();
      _updateState(RecognitionStatus.notRecognised, 0.0);
    }
  }

  bool get shouldTerminateSession {
    if (_notRecognisedSince == null) return false;
    return DateTime.now().difference(_notRecognisedSince!).inSeconds >= 60;
  }

  void _updateState(RecognitionStatus status, double confidence) {
    _bridge.updateState(RecognitionState(
      status: status,
      confidence: confidence,
      timestamp: DateTime.now(),
    ));
  }

  bool _isFaceQualityAcceptable(Face face) {
    if (face.boundingBox.width < 80 || face.boundingBox.height < 80) {
      return false;
    }
    if (face.headEulerAngleY != null && face.headEulerAngleY!.abs() > 30) {
      return false;
    }
    return true;
  }

  InputImage? _convertToInputImage(CameraImage frame) {
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
    } catch (e) {
      return null;
    }
  }

  Future<FaceEmbedding?> _extractEmbedding(
      CameraImage frame, Face face) async {
    if (_interpreter == null) return null;
    if (frame.planes.length == 1) return null;

    try {
      final faceImage = _cropAndResizeFace(frame, face);
      if (faceImage == null) return null;

      final input = _preprocessImage(faceImage);
      final output = List.filled(128, 0.0).reshape([1, 128]);
      _interpreter!.run(input, output);

      return FaceEmbedding(List<double>.from(output[0]));
    } catch (e) {
      return null;
    }
  }

  img.Image? _cropAndResizeFace(CameraImage frame, Face face) {
    try {
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