import 'dart:convert';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import '../recognition/face_embedding.dart';

// Handles persisting and retrieving the enrolled face embedding
// Stores as JSON in secure storage — never as raw images (satisfies NF-04)

class FaceEnrolmentService {
  static const _storageKey = 'enrolled_face_embedding';
  final _storage = const FlutterSecureStorage();

  // Save the averaged embedding from enrolment
  Future<void> saveEmbedding(FaceEmbedding embedding) async {
    final json = jsonEncode(embedding.toJson());
    await _storage.write(key: _storageKey, value: json);
  }

  // Load the stored embedding — returns null if not enrolled yet
  Future<FaceEmbedding?> loadEmbedding() async {
    final json = await _storage.read(key: _storageKey);
    if (json == null) return null;
    return FaceEmbedding.fromJson(jsonDecode(json));
  }

  // Check if a face has been enrolled
  Future<bool> isEnrolled() async {
    final json = await _storage.read(key: _storageKey);
    return json != null;
  }

  // Re-enrolment — overwrites existing data (satisfies FR-06)
  Future<void> clearEnrolment() async {
    await _storage.delete(key: _storageKey);
  }
}